#!/usr/bin/env python
import sys
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Union
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
from pyocto.associator import VelocityModel1D


try:
    from read_yml import read_config
except ModuleNotFoundError:
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

        # read csv file and trim it from start to end
        if self.type == "phasenetsds":
            self.load_phasenetsds()
        elif self.type == "phasenet":
            self.load_phasenetsds()
        else:
            self.load_eqt()

            self.df = self.df[(self.df["time"] >= start) & (self.df["time"] <= end)]

    def load_eqt(self) -> None:
        self.df = pd.read_csv(
            self.filename,
            parse_dates=["p_arrival_time", "s_arrival_time"],
            low_memory=False,
        )
        self.df["phase_time"] = self.df[["p_arrival_time", "s_arrival_time"]].min(
            axis=1
        )
        # FIXME: col rename ?

    def load_phasenet(self) -> None:
        self.df = pd.read_csv(self.filename, low_memory=False)
        # FIXME: col rename ?

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

    def preprocessing(self) -> None:
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
        else:
            self.start = self.df["phase_time"].min()
        if self.end:
            self.df = self.df[self.df["phase_time"] < self.end]
        else:
            self.end = self.df["phase_time"].max()
        if self.df.empty:
            raise ValueError(f"No data in time range [{self.start}, {self.end}].")

        # remove what seems to be fake picks
        if "phase_index" in self.df.columns:
            self.df = self.df[self.df["phase_index"] != 1]

        # limits to 10^-4 seconds same as NLL (needed to unload some picks)
        self.df["phase_time"] = self.df["phase_time"].dt.round("0.0001s")
        self.df.sort_values(by=["phase_time"], inplace=True)

        # get rid off nan value when importing phases without eventid
        self.df = self.df.replace({np.nan: None})

        # keeps only network.station
        self.df["station_id"] = self.df["station_id"].map(
            lambda x: ".".join(x.split(".")[:2])
        )

    def remove_black_listed_stations(self, black_list: List[str]) -> None:
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
    info_sta: Optional[Union[Inventory, str]] = None

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
                self.info_sta = self.inventory
        else:
            if is_valid_url(self.fdsnws_url):
                logger.info(
                    f"Using fdsnws {self.fdsnws_url} to get station coordinates."
                )
                self.info_sta = self.fdsnws_url
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
    time_path: str
    template_path: str
    default_velocity_profile: str
    velocity_profiles: List[NonLinLocVelocityProfile]
    channel_hint: str
    verbose: bool
    enable_scatter: bool
    default_template_file: Optional[str] = None
    min_phase: Optional[int] = -1  # no limit

    def __post_init__(self) -> None:
        if not os.path.exists(self.nlloc_bin):
            raise FileNotFoundError(f"File {self.nlloc_bin} does not exist !")

        if not os.path.exists(self.scat2latlon_bin):
            raise FileNotFoundError(f"File {self.scat2latlon_bin} does not exist !")

        if not os.path.exists(self.channel_hint):
            raise FileNotFoundError(f"File {self.channel_hint} does not exist !")

        if not os.path.isdir(self.time_path):
            raise NotADirectoryError(f"{self.time_path} is not a directory")

        if not os.path.isdir(self.template_path):
            raise NotADirectoryError(f"{self.template_path} is not a directory")

        for profile in self.velocity_profiles:
            # find default velocity profile
            if self.default_velocity_profile == profile.name:
                self.default_template_file = os.path.join(
                    self.template_path, profile.template
                )
            # update
            profile.template_file = os.path.join(self.template_path, profile.template)

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
        OSError: if path can't be created
        PermissionError: if path is not writable
    """

    path: str
    qml_base_filename: str
    event_flush_count: int

    def __post_init__(self) -> None:
        if not os.path.exists(self.path) or not os.path.isdir(self.path):
            try:
                os.makedirs(self.path)
            except OSError as e:
                raise e(f"Can't create directory {self.path}")

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
    current_model: Optional[Model] = None
    travel_time_grid_filename: Optional[str] = None
    velocity_model: Optional[VelocityModel1D] = None

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

        # create travel time grid file
        os.makedirs(self.path, exist_ok=True)
        self.travel_time_grid_filename = os.path.join(
            self.path, self.default_model_name
        )

        self.create_travel_time_grid_file(
            self.current_model.velocity_model, self.travel_time_grid_filename
        )

        # Create 1D velocity model
        self.velocity_model = self.create_velocity_model()

    def create_velocity_model(self) -> VelocityModel1D:
        tolerance = self.current_model.velocity_model.tolerance
        velocity_model = VelocityModel1D(
            path=self.travel_time_grid_filename,
            tolerance=tolerance,
            # association_cutoff_distance=None,
            # location_cutoff_distance=None,
            # surface_p_velocity=None,
            # surface_s_velocity=None,
        )
        return velocity_model

    def create_travel_time_grid_file(
        self, vmodel: VelocityModel, filename: str
    ) -> None:
        # create dataframe
        profil_model = pd.DataFrame(
            {
                "depth": vmodel.depth,
                "vp": vmodel.vp,
                "vs": vmodel.vs,
            }
        )

        # create travel time grid
        VelocityModel1D.create_model(
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

    # NLL will discard any location with number of phase < min_phase
    # take into account cluster parameters to set it accordingly
    # use -1 to not set a limit
    myconf.nll.min_phase = (
        myconf.cluster.min_station_count + myconf.cluster.min_station_with_P_and_S
    )

    # Finalize zones
    myconf.zones.load_zones(myconf.nll)

    # load picks
    # myconf.pick.load_phasenetsds()
    myconf.pick.remove_black_listed_stations(myconf.station.blacklist)
    myconf.pick.preprocessing()
    return myconf


if __name__ == "__main__":
    try:
        from localization import NllLoc
    except ModuleNotFoundError:
        from dbclust.localization import NllLoc

    try:
        from clusterize import Clusterize
    except ModuleNotFoundError:
        from dbclust.clusterize import Clusterize

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

    locator = NllLoc(
        myconf.nll.nlloc_bin,
        myconf.nll.scat2latlon_bin,
        myconf.nll.time_path,
        myconf.nll.template_path,
        nll_channel_hint=myconf.nll.channel_hint,
        tmpdir=myconf.file.tmp_path,
        double_pass=myconf.relocation.double_pass,
        P_time_residual_threshold=myconf.relocation.P_time_residual_threshold,
        S_time_residual_threshold=myconf.relocation.S_time_residual_threshold,
        dist_km_cutoff=myconf.relocation.dist_km_cutoff,
        nll_min_phase=myconf.nll.min_phase,
        min_station_with_P_and_S=myconf.cluster.min_station_with_P_and_S,
        quakeml_settings=asdict(myconf.quakeml),
        nll_verbose=myconf.nll.verbose,
        keep_scat=myconf.nll.enable_scatter,
        log_level=logger.level,
    )
    ic(vars(locator))

    myclust = Clusterize(
        phases=None,  # empty cluster / constructor only
        average_velocity=myconf.cluster.average_velocity,
        min_station_count=myconf.cluster.min_station_count,
        min_station_with_P_and_S=myconf.cluster.min_station_with_P_and_S,
        max_search_dist=myconf.cluster.max_search_dist,
        P_uncertainty=myconf.pick.P_uncertainty,
        S_uncertainty=myconf.pick.S_uncertainty,
        tt_matrix_fname=myconf.cluster.pre_computed_tt_matrix_file,
        tt_matrix_save=myconf.cluster.tt_matrix_save,
        zones=myconf.zones.zones,
        log_level=logger.level,
    )
    ic(vars(myclust))
