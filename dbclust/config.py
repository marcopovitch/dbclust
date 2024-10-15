#!/usr/bin/env python
import argparse
import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from datetime import datetime
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import pyproj
from dacite import from_dict
from db import duckdb_init
from icecream import ic
from obspy import Inventory
from obspy import read_inventory
from pyocto.associator import VelocityModel1D
from read_yml import read_config
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon

# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("dbclust_config")
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
    automatic_cleanup_tmp: bool

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
    filename: Union[str, None]
    type: Union[str, None]
    P_uncertainty: float
    S_uncertainty: float
    P_proba_threshold: float
    S_proba_threshold: float
    start: Optional[Union[datetime, pd.Timestamp]] = None
    end: Optional[Union[datetime, pd.Timestamp]] = None
    df: Optional[pd.DataFrame] = None

    def __post_init__(self) -> None:
        if self.type not in ["csv", "parquet", None]:
            raise ValueError(f"Pick file format {self.type} is not recognized !")

        if not self.type:
            return

        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"File {self.filename} does not exist !")

        if not os.access(self.filename, os.R_OK):
            raise PermissionError(f"{self.filename}.")

        if self.start:
            self.start = pd.to_datetime(self.start, utc=True).to_datetime64()

        if self.end:
            self.end = pd.to_datetime(self.end, utc=True).to_datetime64()

        # Check parquet or csv file
        if self.type == "parquet":
            try:
                table = pq.read_table(
                    self.filename, columns=[], use_pandas_metadata=False
                )
            except:
                raise ValueError(f"{self.filename} is not parquet formated !")
            if os.path.isdir(self.filename):
                self.filename = os.path.join(self.filename, "**", "*.parquet")
        else:
            # CSV
            try:
                with open(self.filename, "r") as file:
                    first_line = file.readline().strip()
                    nbcol = first_line.count(",")
                    if nbcol < 4:
                        raise ValueError(
                            f"{self.filename} is not a csv file or some columns are missing ({nbcol}) !"
                        )
            except Exception as e:
                raise e

        # set min, max time from data
        conn = duckdb_init(self.filename, self.type)
        rqt = "SELECT MIN(phase_time), MAX(phase_time) FROM PICKS"
        min, max = conn.sql(rqt).fetchall().pop()
        if not self.start:
            self.start = min
        if not self.end:
            self.end = max


@dataclass
class FdsnConfig:
    """Manage different FDSN web services

    Attributes:
        debug (str): enable debug mode
        default(str): default FDSN web service URL
        url (str): FDSN web service URL dictionary

    Raises:
        URLError: if URL is not valid
    """

    default: str
    hosts: Dict[str, str]
    url: Optional[str] = None

    def __post_init__(self) -> None:
        # check url validity for each service
        for key, value in self.hosts.items():
            if not is_valid_url(value, syntax_only=True):
                raise URLError(f"{key} URL {value} is not valid !")
        self.url = self.hosts[self.default]

    def set_url_from_service_name(self, service: str) -> None:
        if service not in self.hosts:
            raise ValueError(f"Service {service} not found in FDSN hosts !")
        self.url = self.hosts[service]

    def get_url(self) -> str:
        return self.url


@dataclass
class StationConfig:
    """Manage how to get stations coordinates.

    This class provides a configuration for fetching station coordinates.
    It allows fetching coordinates either from an inventory file or from a FDSN web service.

    Attributes:
        fetch_method (str):
            The method used to fetch station coordinates. Should be either "inventory" or "fdsnws".
        fdsnws_url (Optional[str]):
            The URL of the FDSN web service. Required if fetch_method is "fdsnws".
        inventory_files (Optional[List[str]]):
            A list of inventory file paths. Required if fetch_method is "inventory".
        blacklist (Optional[List[str]]):
            A list of station codes to be excluded from the fetched coordinates.
        rename (Optional[List[dict]]):
            A list of dictionaries specifying station code renaming rules.
        frequency_threshold (Optional[float]):
            A threshold value for filtering stations based on their frequency.
        inventory (Optional[Inventory]):
            An instance of the `Inventory` class containing station information.
        info_sta (Optional[Union[Inventory, str]]):
            Information about the stations, either an `Inventory` instance or a URL.

    Raises:
        ValueError: If fetch_method is not "inventory" or "fdsnws".
        URLError: If fdsnws_url is not a valid URL or cannot be joined.

    Methods:
        __post_init__: A method automatically called after the object is initialized.

    """

    fetch_method: str
    fdsnws_url: Optional[str] = None
    fdsnws: Optional[FdsnConfig] = None
    inventory_files: Optional[List[str]] = None
    blacklist: Optional[List[str]] = None
    rename: Optional[dict] = None
    frequency_threshold: Optional[float] = None
    inventory: Optional[Inventory] = None
    info_sta: Optional[Union[Inventory, str]] = None

    def __post_init__(self) -> None:
        if self.fetch_method not in ["inventory", "fdsnws"]:
            raise ValueError("Invalid fetch_method: should be 'inventory' or 'fdsnws'!")

        if self.fetch_method == "inventory":
            self.inventory = Inventory()
            for f in self.inventory_files:
                logger.info(f"Reading inventory file {f}")
                self.inventory.extend(read_inventory(f))
                self.info_sta = self.inventory
        else:
            logger.debug(f"Using fdsnws {self.fdsnws.url} to get station coordinates.")
            self.info_sta = self.fdsnws.get_url()


@dataclass
class TimeConfig:
    """
    Configuration class for time settings.
    """

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
    verbose: bool
    enable_scatter: bool
    default_template_file: Optional[str] = None
    min_phase: Optional[int] = -1  # no limit

    def __post_init__(self) -> None:
        if not os.path.exists(self.nlloc_bin):
            raise FileNotFoundError(f"File {self.nlloc_bin} does not exist !")

        if not os.path.exists(self.scat2latlon_bin):
            raise FileNotFoundError(f"File {self.scat2latlon_bin} does not exist !")

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
    P_time_residual_threshold: Union[float, None]
    S_time_residual_threshold: Union[float, None]
    double_pass: bool
    keep_manual_picks: bool
    use_deactivated_arrivals: bool
    use_pick_zone: bool
    dist_km_cutoff: Optional[float] = None

    # enable pick relabeling based on pick zone and score threshold
    # supersed P and S time residual threshold
    # the pick zone is defined in the zones section
    use_pick_zone: Optional[bool] = False
    # only used if use_pick_zone is True
    # relabel pick if score is above this threshold
    min_score_threshold_pick_zone: Optional[float] = 1
    enable_relabel_pick_zone: Optional[bool] = False
    # remove outliers from pick zone
    enable_cleanup_pick_zone: Optional[bool] = False



@dataclass
class QuakemlConfig:
    event_prefix: str
    smi_base: str
    agency_id: str
    author: str
    evaluation_mode: str
    method_id: str
    model_id: Optional[str] = None


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
    polygon: List[List[float]]
    picks_delimiter: List[Dict[str, List[List[float]]]] = field(default_factory=list)
    mu: Optional[List[Dict[str, float]]] = None
    sigma: Optional[List[Dict[str, float]]] = None

    def __str__(self) -> str:
        txt = f"zone:\n\tname: '{self.name}'\n\tprofile: '{self.velocity_profile}'\n\tpolygon: {self.polygon}"
        txt += "\n\tpicks_delimiter:"
        for item in self.picks_delimiter:
            for key, value in item.items():
                txt += f"\n\t\t{key}: {value}"

        return txt


@dataclass
class Zones:
    zones: List[Zone]
    polygons: Optional[gpd.GeoDataFrame] = None

    def load_zones(self, nll_cfg: NonLinLocConfig) -> None:
        records = []
        for z in self.zones:
            # sanity check
            found = False
            for vp in nll_cfg.velocity_profiles:
                if z.velocity_profile == vp.name:
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"Can't find zone velocity profile {z.velocity_profile}"
                )

            # create shapely polygon zone from list of coordinates
            polygon = Polygon(z.polygon)

            # get picks_delimiter polygons
            gdf_pick_delimiter = gpd.GeoDataFrame()
            if z.picks_delimiter:
                picks_delimiter_polygons = []
                names = []  # pick family name (Pn, Pg, Sn, Sg, ...)

                # iterate over Pg, Pn, Sg, Sn, ...
                for item in z.picks_delimiter:
                    for key, value in item.items():
                        picks_delimiter_polygons.append(Polygon(value))
                        names.append(key)

                df = pd.DataFrame({"name": names, "geometry": picks_delimiter_polygons})
                gdf = gpd.GeoDataFrame(df, geometry="geometry")
                gdf["region"] = z.name
                gdf["mu"] = z.mu
                gdf["sigma"] = z.sigma
                gdf_pick_delimiter = pd.concat(
                    [gdf_pick_delimiter, gdf],
                    ignore_index=True,
                )

            records.append(
                {
                    "name": z.name,
                    "velocity_profile": vp.name,
                    "template": vp.template_file,
                    "geometry": polygon,
                    "mu": z.mu,
                    "sigma": z.sigma,
                    "picks_delimiter": gdf_pick_delimiter,
                }
            )

        if not len(records):
            raise ValueError(f"Zones defined ... but empty !")

        self.polygons = gpd.GeoDataFrame(records)

    def get_velocity_profile_name(self, zone_name: str) -> str:
        """Get velocity profile name from zone name

        Args:
            zone_name (str): zone name

        Returns:
            str: velocity profile name
        """
        for zone in self.zones:
            if zone.name == zone_name:
                return zone.velocity_profile
        return ""

    def get_zone_from_name(self, name: str) -> gpd.GeoDataFrame:
        """Get zone given it's name

        Args:
            name (str): zone name to get

        Returns:
            gpd.GeoDataFrame: zone dataframe
        """
        for index, row in self.polygons.iterrows():
            if row["name"] == name:
                return row
        return gpd.GeoDataFrame()

    def find_zone(
        self, latitude: float = None, longitude: float = None
    ) -> Tuple[gpd.GeoDataFrame, float]:
        """Find zone

        Args:
            latitude (float, optional): Defaults to None.
            longitude (float, optional): Defaults to None.
            zones (gpd.GeoDataFrame, optional): Defaults to None.

        Returns:
            gpd.GeoDataFrame: geodataframe found or an empty one if nothing found.
        """
        point_shapely = Point(longitude, latitude)
        for index, row in self.polygons.iterrows():

            polygon = row["geometry"]
            if polygon.contains(point_shapely):

                # convert wgs84 coord to lambert II (metric)
                transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:27572")
                x_point, y_point = transformer.transform(longitude, latitude)
                sommets_lambert = [
                    transformer.transform(lon, lat)
                    for lon, lat in polygon.exterior.coords
                ]

                polygon = Polygon(sommets_lambert)
                point_shapely = Point(x_point, y_point)

                lines = [
                    LineString(
                        [
                            polygon.exterior.coords[i],
                            polygon.exterior.coords[
                                (i + 1) % len(polygon.exterior.coords)
                            ],
                        ]
                    )
                    for i in range(len(polygon.exterior.coords))
                ]
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=RuntimeWarning)
                    try:
                        distances = [point_shapely.distance(line) for line in lines]
                    except RuntimeWarning as rw:
                        logger.debug(f"Can't compute distance(point, polygon): {row}")
                        distance_km = None
                    else:
                        distance_km = min(distances) / 1000
                return row, distance_km
        return gpd.GeoDataFrame(), None

    def show_zones(self):
        print("Zone name: velocity profile")
        for index, row in self.polygons.iterrows():
            print(f'\t{row["name"]}: {row["velocity_profile"]}')
        print()


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


@dataclass
class ParallelConfig:
    n_workers: int = None
    # Use Timedelta unit
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.html
    partition_duration: str = "1D"
    nb_partitions: Optional[int] = None
    time_partitions: Optional[List] = None

    def __post_init__(self):
        if not self.n_workers:
            self.n_workers = os.cpu_count()

    def get_time_partitions(self, time_cfg: TimeConfig, pick_cfg: PickConfig) -> List:
        # ic(pick_cfg.start, pick_cfg.end)
        nb_periods = self.get_nb_of_divisions(
            pick_cfg.start, pick_cfg.end, self.partition_duration
        )
        time_divisions = (
            pd.date_range(
                start=pick_cfg.start,
                end=pick_cfg.end,
                periods=nb_periods,
                inclusive="both",
            )
            .to_series()
            .to_list()
        )

        if nb_periods == 1:
            # only one period: use [start, end] without overlap
            adjusted_time_divisions = [[pick_cfg.start, pick_cfg.end]]
        else:
            adjusted_time_divisions = [
                (start, end + pd.Timedelta(seconds=time_cfg.overlap_window))
                for start, end in zip(time_divisions, time_divisions[1:])
            ]

        self.nb_partitions = len(adjusted_time_divisions)

        return adjusted_time_divisions

    @staticmethod
    def get_nb_of_divisions(start, end, freq):
        return math.ceil((end - start) / pd.Timedelta(freq))


@dataclass
class DBClustConfig:
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
    parallel: ParallelConfig
    fdsnws_event: FdsnConfig

    def __init__(self, filename, config_type="std") -> None:
        self.filename = filename
        logger.info(filename)
        self.config_type = config_type

        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"File {self.filename} does not exist !")
        self.yaml_data = read_config(self.filename)

        for key, data_class in self.__annotations__.items():
            if self.config_type == "reloc" and key in [
                "parallel",
                "time",
                "cluster",
                "pyocto",
                "zones",
                "catalog",
            ]:
                logger.warning(f"Missing section '{key}' in yaml file !")
                continue
            else:
                if key not in self.yaml_data.keys():
                    raise ValueError(f"Missing section '{key}' in yaml file !")
                setattr(
                    self,
                    key,
                    from_dict(data_class=data_class, data=self.yaml_data[key]),
                )
        if config_type == "reloc":
            return

        # NLL will discard any location with number of phase < min_phase
        # take into account cluster parameters to set it accordingly
        # use -1 to not set a limit
        self.nll.min_phase = (
            self.cluster.min_station_count + self.cluster.min_station_with_P_and_S
        )

        # parallel
        self.parallel.time_partitions = self.parallel.get_time_partitions(
            self.time, self.pick
        )

        # Finalize zones
        self.zones.load_zones(self.nll)

        # Set default velocity model
        self.quakeml.model_id = self.nll.default_velocity_profile
        # ic(self.quakeml)
        assert len(self.parallel.time_partitions)

    def show(self):
        # debug
        for key, value in self.__annotations__.items():
            if self.config_type == "reloc" and key in [
                "parallel",
                "time",
                "cluster",
                "pyocto",
                "zones",
                "catalog",
            ]:
                continue
            attribute_value = getattr(self, key)
            ic(key, attribute_value)

        # test zone
        ic(self.pick.df)


def is_valid_url(url: str, syntax_only: bool = False) -> bool:
    """Check if url syntax is valid, and url is joinable

    Args:
        url (str): url string

    Returns:
        bool
    """
    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme and parsed_url.netloc:
            if syntax_only:
                return True
            with urlopen(url):
                pass
            return True
    except URLError:
        pass

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--conf_file",
        default=None,
        dest="config_file",
        help="yaml configuration file.",
        type=str,
    )

    parser.add_argument(
        "-t",
        "--conf_type",
        default=None,
        dest="config_type",
        help="std|reloc configuration type.",
        type=str,
    )

    args = parser.parse_args()
    if not args.config_file:
        parser.print_help()
        sys.exit(255)

    myconf = DBClustConfig(args.config_file, config_type=args.config_type)
    myconf.show()
