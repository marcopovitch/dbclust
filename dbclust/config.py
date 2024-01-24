#!/usr/bin/env python
import sys
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Type, Optional, List, Union
import logging
from urllib.parse import urlparse
from urllib.request import urlopen
from urllib.error import URLError
from dacite import from_dict
from icecream import ic
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
from obspy import Inventory, read_inventory
import pyocto

try:
    from read_yml import read_config
except:
    from dbclust.read_yml import read_config

# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("config")
logger.setLevel(logging.INFO)


@dataclass
class FilesConfig:
    """Check data path and create temporary directories

    Raises:
        NotADirectoryError: if data_path is not a directory
        e: returns os.makedirs() exception
    """

    data_path: str
    tmp_path: str
    obs_path: str

    def __post_init__(self):
        if not os.path.isdir(self.data_path):
            raise NotADirectoryError(f"{self.data_path} is not a directory.")

        for dir in [self.tmp_path, self.obs_path]:
            try:
                os.makedirs(dir, exist_ok=True)
            except Exception as e:
                raise e(f"{dir}")


@dataclass
class PickConfig:
    """Manages and checks parameters associated with picks

    Raises:
        FileNotFoundError: if pick file is not found
        PermissionError: if pick file is not readable
        ValueError: if file type/format is not recognized
    """

    # path: str
    filename: str
    type: str
    P_uncertainty: float
    S_uncertainty: float
    P_proba_threshold: float
    S_proba_threshold: float
    start: Optional[Union[datetime, pd.Timestamp]] = None
    end: Optional[Union[datetime, pd.Timestamp]] = None
    df: Optional[pd.DataFrame] = None

    def __post_init__(self) -> None:
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"File {self.filename} does not exist !")

        if not os.access(self.filename, os.R_OK):
            raise PermissionError(f"{self.filename}.")

        if self.type not in ["eqt", "phasenet", "phasenetsds"]:
            raise ValueError(f"Pick format {self.type} is not recognized !")

        if self.start:
            self.start = pd.to_datetime(self.start, utc=True)

        if self.end:
            self.end = pd.to_datetime(self.end, utc=True)

    def load_eqt(self) -> None:
        self.df = pd.read_csv(
            self.filename,
            parse_dates=["p_arrival_time", "s_arrival_time"],
            low_memory=False,
        )
        self.df["phase_time"] = self.df[["p_arrival_time", "s_arrival_time"]].min(
            axis=1
        )

    def load_phasenet(self) -> None:
        self.df = pd.read_csv(self.filename, low_memory=False)

    def load_phasenetsds(self) -> None:
        self.df = pd.read_csv(self.filename, low_memory=False)
        self.df.rename(
            columns={
                "seedid": "station_id",
                "phasename": "phase_type",
                "time": "phase_time",
                "probability": "phase_score",
            },
            inplace=True,
        )

    def prepocessing(self) -> None:
        """Preprocess picks

        - get only picks from [start, end]
        - limits pick time to 10^-4 seconds
        - remove fake picks
        - remove black listed stations
        - get rid off nan value
        - keeps only network.station

        Raises:
            ValueError: if self.df is empty
        """
        self.df["phase_time"] = pd.to_datetime(self.df["phase_time"], utc=True)
        if self.start:
            self.df = self.df[self.df["phase_time"] >= self.start]
        if self.end:
            self.df = self.df[self.df["phase_time"] < self.end]
        if self.df.empty:
            raise ValueError(f"No data in time range [{self.start}, {self.end}].")

        # remove what seems to be fake picks
        if "phase_index" in self.df.columns:
            self.df = self.df[self.df["phase_index"] != 1]

        # limits to 10^-4 seconds same as NLL (needed to unload some picks)
        self.df["phase_time"] = self.df["phase_time"].dt.round("0.0001S")
        self.df.sort_values(by=["phase_time"], inplace=True)

        # get rid off nan value when importing phases without eventid
        self.df = self.df.replace({np.nan: None})

        # keeps only network.station
        self.df["station_id"] = self.df["station_id"].map(
            lambda x: ".".join(x.split(".")[:2])
        )

    def remove_black_listed_stations(self, black_list) -> None:
        logger.info("Removing black listed channels:")
        if not black_list:
            logger.info("\t- black_list is empty")
            return

        for b in black_list:
            before = len(self.df)
            self.df = self.df[~self.df["station_id"].str.contains(b, regex=True)]
            after = len(self.df)
            logger.info(f"\t- removed {before-after} {b} picks")


@dataclass
class StationConfig:
    """Manage how to get stations coordinates

    Raises:
        ValueError: allow only "inventory" or "fdsnws" to get stations coordinates
        URLError: when fdsnws_url is not valid or is not joinable
    """

    fetch_method: str
    fdsnws_url: Optional[str] = None
    inventory_files: Optional[List[str]] = None
    blacklist: Optional[List[str]] = None
    inventory: Optional[Inventory] = None

    def __post_init__(self) -> None:
        if self.fetch_method not in ["inventory", "fdsnws"]:
            raise ValueError(
                "Invalid fetch_method: should be 'inventory' or 'fdsnws' !"
            )

        if self.fetch_method == "inventory":
            self.inventory = Inventory()
            for f in self.inventory_files:
                logger.info(f"Reading inventory file {f}")
                self.inventory.extend(read_inventory(f))
        else:
            if is_valid_url(self.fdsnws_url):
                logger.info(
                    f"Using fdsnws {self.fdsnws_url} to get station coordinates."
                )
            else:
                raise URLError(f"{self.fdsnws_url}")


@dataclass
class TimeConfig:
    # window_length: int
    # overlap_length: int
    time_window: float = 7  # minutes
    overlap_window: float = 40  # seconds


@dataclass
class ClusterConfig:
    """Manages and checks parameters associated with cluster

    Raises:
        FileExistsError: if pre_computed_tt_matrix_file already exists
    """

    min_cluster_size: int
    min_station_count: int
    min_station_with_P_and_S: int
    average_velocity: float
    min_picks_common: int
    max_search_dist: Optional[float] = 0.0
    pre_computed_tt_matrix_file: Optional[str] = None

    def __post_init__(self) -> None:
        if self.pre_computed_tt_matrix_file:
            self.tt_matrix_save = True
            if os.path.exists(self.pre_computed_tt_matrix_file):
                raise FileExistsError(
                    f"File {self.pre_computed_tt_matrix_file} already exists !"
                )
        else:
            self.tt_matrix_save = False


@dataclass
class NonLinLocVelocityProfile:
    name: str
    template: str
    template_file: Optional[str] = None


@dataclass
class NonLinLocConfig:
    """Manages and checks parameters associated with NonLinLoc

    Raises:
        FileNotFoundError: when binaries, template can't be found
        NotADirectoryError: when time or template dir do not exist
        ValueError: when default_velocity_profile is not defined
    """

    nlloc_bin: str
    scat2latlon_bin: str
    nll_time_path: str
    nll_template_path: str
    default_velocity_profile: str
    velocity_profiles: List[NonLinLocVelocityProfile]
    nll_channel_hint: str
    verbose: bool
    enable_scatter: bool
    default_template_file: Optional[str] = None

    def __post_init__(self) -> None:
        if not os.path.exists(self.nlloc_bin):
            raise FileNotFoundError(f"File {self.nlloc_bin} does not exist !")

        if not os.path.exists(self.scat2latlon_bin):
            raise FileNotFoundError(f"File {self.scat2latlon_bin} does not exist !")

        if not os.path.exists(self.nll_channel_hint):
            raise FileNotFoundError(f"File {self.nll_channel_hint} does not exist !")

        if not os.path.isdir(self.nll_time_path):
            raise NotADirectoryError(f"{self.nll_time_path} is not a directory")

        if not os.path.isdir(self.nll_template_path):
            raise NotADirectoryError(f"{self.nll_template_path} is not a directory")

        for profile in self.velocity_profiles:
            # find default velocity profile
            if self.default_velocity_profile == profile.name:
                self.default_template_file = os.path.join(
                    self.nll_template_path, profile.template
                )
            # update
            profile.template_file = os.path.join(
                self.nll_template_path, profile.template
            )

        if not self.default_template_file:
            raise ValueError(
                f"Referenced template {self.default_velocity_profile} is not defined !"
            )

        if not os.path.exists(self.default_template_file):
            raise FileNotFoundError(
                f"File {self.default_template_file} does not exist !"
            )


@dataclass
class RelocationConfig:
    double_pass: bool
    P_time_residual_threshold: float
    S_time_residual_threshold: float
    dist_km_cutoff: Optional[float] = None


@dataclass
class QuakemlConfig:
    event_prefix: str
    smi_base: str
    agency_id: str
    author: str
    evaluation_mode: str
    method_id: str
    # model_id: Optional[str] = None


@dataclass
class CatalogConfig:
    """Manages and checks parameters associated with catalog writing

    Raises:
        NotADirectoryError: if path is not a directory
        PermissionError: if path is not writable
    """

    path: str
    qml_base_filename: str
    event_flush_count: int

    def __post_init__(self) -> None:
        if not os.path.isdir(self.path):
            raise NotADirectoryError(f"{self.path} is not a directory")

        if not os.access(self.path, os.W_OK):
            raise PermissionError(f"Can't write in {self.path} directory.")


@dataclass
class Zone:
    name: str
    velocity_profile: str
    polygone: List[List[float]]


@dataclass
class Zones:
    zones: List[Zone]
    polygones: Optional[gpd.geodataframe.GeoDataFrame] = field(default=None)

    def load_zones(self, nll_cfg: NonLinLocConfig) -> None:
        records = []
        for z in self.zones:
            found = False
            for vp in nll_cfg.velocity_profiles:
                if z.velocity_profile == vp.name:
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"Can't find zone velocity profile {z.velocity_profile}"
                )

            polygon = Polygon(z.polygone)
            records.append(
                {
                    "name": z.name,
                    "velocity_profile": vp.name,
                    "template": vp.template_file,
                    "geometry": polygon,
                }
            )
        if not len(records):
            raise ValueError(f"Zones defined ... but empty !")

        self.polygones = gpd.GeoDataFrame(records)

    def find_zone(self, latitude: float = None, longitude: float = None) -> gpd:
        """Find zone

        Args:
            latitude (float, optional): Defaults to None.
            longitude (float, optional): Defaults to None.
            zones (gpd.GeoDataFrame, optional): Defaults to None.

        Returns:
            gpd.GeoDataFrame: geodataframe found or an empty one if nothing found.
        """
        point_shapely = Point(longitude, latitude)
        for index, row in self.polygones.iterrows():
            if row["geometry"].contains(point_shapely):
                return row
        return gpd.GeoDataFrame()


@dataclass
class Associator:
    time_before: float
    max_pick_overlap: float
    min_pick_fraction: float
    zlim: List[float]
    min_node_size: float
    min_node_size_location: float
    pick_match_tolerance: float
    n_picks: int
    n_p_picks: int
    n_s_picks: int
    n_p_and_s_picks: int


@dataclass
class VelocityModel:
    depth: List[float]
    vp: List[float]
    vs: List[float]
    tolerance: float
    grid_spacing_km: float
    max_horizontal_dist_km: float
    max_vertical_dist_km: float


@dataclass
class Model:
    name: str
    associator: Associator
    velocity_model: VelocityModel


@dataclass
class PyoctoConfig:
    """Manages and checks consistency pyocto parameters

    Raises:
        ValueError: when no model name exists
    """

    default_model_name: str
    path: str
    models: List[Model]
    # associator and velocity_model are in current_model.keys()
    current_model: Optional[dict] = None
    model_filename: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.default_model_name:
            self.current_model = None
            return

        for model in self.models:
            if model.name == self.default_model_name:
                self.current_model = model
                break

        if not self.current_model:
            raise ValueError(
                f"Referenced model {self.default_model_name} is not defined !"
            )

        # create velocity model
        os.makedirs(self.path, exist_ok=True)
        self.model_filename = os.path.join(self.path, self.default_model_name)
        self.create_velocity_model_file(
            self.current_model.velocity_model, self.model_filename
        )

    def create_velocity_model_file(self, vmodel: VelocityModel, filename: str) -> None:
        profil_model = pd.DataFrame(
            {
                "depth": vmodel.depth,
                "vp": vmodel.vp,
                "vs": vmodel.vs,
            }
        )

        pyocto.VelocityModel1D.create_model(
            profil_model,
            vmodel.grid_spacing_km,
            vmodel.max_horizontal_dist_km,
            vmodel.max_vertical_dist_km,
            filename,
        )


class Config:
    file: FilesConfig
    pick: PickConfig
    station: StationConfig
    time: TimeConfig
    cluster: ClusterConfig
    nll: NonLinLocConfig
    relocation: RelocationConfig
    quakeml: QuakemlConfig
    catalog: CatalogConfig
    pyocto: PyoctoConfig
    zones: Zones


def is_valid_url(url: str) -> bool:
    """Check if url syntax is valid, and url is joinable

    Args:
        url (str): url string

    Returns:
        bool
    """
    try:
        # Analyse de l'URL
        parsed_url = urlparse(url)

        # Vérification si l'URL a un schéma et un netloc (domaine)
        if parsed_url.scheme and parsed_url.netloc:
            # Tentative d'ouverture de l'URL pour vérifier son existence
            with urlopen(url):
                pass
            return True
    except URLError:
        pass

    return False


def get_config_from_file(yaml_file: str, verbose: bool = False) -> Config:
    yaml_data = read_config(yaml_file)

    myconf = Config()
    for key, data_class in myconf.__annotations__.items():
        if key not in yaml_data.keys():
            raise ValueError(f"Missing section '{key}' in yaml file !")
        setattr(myconf, key, from_dict(data_class=data_class, data=yaml_data[key]))

    # Finalize zones
    myconf.zones.load_zones(myconf.nll)

    # load picks
    myconf.pick.load_phasenetsds()
    myconf.pick.remove_black_listed_stations(myconf.station.blacklist)
    myconf.pick.prepocessing()
    return myconf


if __name__ == "__main__":
    myconf = get_config_from_file(
        "/Users/marc/Data/DBClust/selestat/dbclust-selestat-mod.yml", verbose=True
    )

    # debug
    for key, value in myconf.__annotations__.items():
        attribute_value = getattr(myconf, key)
        ic(key, attribute_value)

    # test zone
    ic(myconf.zones.find_zone(48, 7))
    ic(myconf.pick.df)

