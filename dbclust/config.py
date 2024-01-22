#!/usr/bin/env python
import sys
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List
import logging
from urllib.parse import urlparse
from urllib.request import urlopen
from urllib.error import URLError
from dacite import from_dict
from icecream import ic
import geopandas as gpd
from shapely.geometry import Polygon, Point
from obspy import Inventory, read_inventory

try:
    from read_yml import read_config
except:
    from dbclust.read_yml import read_config

try:
    from zones import load_zones
except:
    from dbclust.zones import load_zones

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

    def __post_init__(self):
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"File {self.filename} does not exist !")

        if not os.access(self.filename, os.R_OK):
            raise PermissionError(f"{self.filename}.")

        if self.type not in ["eqt", "phasenet"]:
            raise ValueError(f"Pick format {self.type} is not recognized !")


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

    def __post_init__(self):
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
    start: Optional[datetime] = None
    end: Optional[datetime] = None


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

    def __post_init__(self):
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

    def __post_init__(self):
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
            if self.default_velocity_profile == profile.name:
                self.default_template_file = os.path.join(
                    self.nll_template_path, profile.template
                )
                break

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

    def __post_init__(self):
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

    def load_zones(self, nll_cfg: NonLinLocConfig):
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
        self.polygones = gpd.GeoDataFrame(records)


@dataclass
class PyoctoAssociatorConfig:
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
class PyoctoVelocityModelConfig:
    depth: List[float]
    vp: List[float]
    vs: List[float]
    tolerance: float
    grid_spacing_km: float
    max_horizontal_dist_km: float
    max_vertical_dist_km: float


@dataclass
class PyoctoConfig:
    """Check consistency parameters

    Raises:
        ValueError: when no model name exists
    """

    default_model_name: str
    models: List[dict]
    current_model: Optional[dict] = None

    def __post_init__(self):
        if not self.default_model_name:
            self.current_model = None
            return

        for model in self.models:
            if model["name"] == self.default_model_name:
                self.current_model = model
                return

        raise ValueError(f"Referenced model {self.default_model_name} is not defined !")


def is_valid_url(url):
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


def get_config_from_file(yaml_file: str, verbose: bool = False) -> dict:
    yaml_data = read_config(yaml_file)

    class_mapping = {
        "file": FilesConfig,
        "pick": PickConfig,
        "station": StationConfig,
        "time": TimeConfig,
        "cluster": ClusterConfig,
        "nll": NonLinLocConfig,
        "relocation": RelocationConfig,
        "quakeml": QuakemlConfig,
        "catalog": CatalogConfig,
        "pyocto": PyoctoConfig,
        "zones": Zones,
    }

    myconf = {}
    for key, data_class in class_mapping.items():
        if key not in yaml_data.keys():
            raise ValueError(f"Missing section '{key}' in yaml file !")

        myconf[key] = from_dict(
            data_class=data_class,
            data=yaml_data[key],
        )

    if verbose:
        for key, obj in myconf.items():
            ic(obj)

        pyocto_cfg = myconf["pyocto"]
        ic(pyocto_cfg.current_model)

    nll_cfg = myconf["nll"]
    zones = myconf["zones"]
    zones.load_zones(nll_cfg)
    ic(zones)
    return myconf


if __name__ == "__main__":
    get_config_from_file(
        "/Users/marc/Data/DBClust/selestat/dbclust-selestat-mod.yml", verbose=True
    )
