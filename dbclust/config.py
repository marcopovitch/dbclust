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

import fastparquet
import geopandas as gpd
import pandas as pd
import pyproj
from dacite import from_dict
from icecream import ic
from obspy import Inventory
from obspy import read_inventory
from obspy import UTCDateTime
from pyocto.associator import VelocityModel1D
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon

from dbclust.db import duckdb_init
from dbclust.db import duckdb_init_parquet
from dbclust.read_yml import read_config


# Configure icecream to flush output immediately (needed for Parsl)
def _ic_output(s):
    sys.stderr.write(s + "\n")
    sys.stderr.flush()


ic.configureOutput(outputFunction=_ic_output)

# default logger (uses hierarchical name for selective level control)
logger = logging.getLogger("dbclust.config")


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
            except OSError as e:
                raise OSError(f"Cannot create directory {dir}: {e}") from e


@dataclass
class PickConfig:
    """Manages and checks parameters associated with picks

    Raises:
        FileNotFoundError: if pick file is not found
        PermissionError: if pick file is not readable
        ValueError: if file type/format is not recognized
    """

    # path: str
    filenames: List[str]
    type: Union[str, None]
    P_uncertainty: float
    S_uncertainty: float
    P_proba_threshold: float
    S_proba_threshold: float
    P_proximity_threshold: float
    S_proximity_threshold: float
    start: Optional[Union[datetime, pd.Timestamp]] = None
    end: Optional[Union[datetime, pd.Timestamp]] = None
    df: Optional[pd.DataFrame] = None
    # Geographic extent filter (optional)
    # bbox: {"min_lat": float, "max_lat": float, "min_lon": float, "max_lon": float}
    bbox: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        # Validate geographic extent parameters
        if self.bbox:
            required_keys = {"min_lat", "max_lat", "min_lon", "max_lon"}
            missing = required_keys - set(self.bbox.keys())
            if missing:
                raise ValueError(
                    f"bbox is missing required keys: {', '.join(sorted(missing))}. "
                    f"Required: min_lat, max_lat, min_lon, max_lon"
                )
            if self.bbox["min_lat"] >= self.bbox["max_lat"]:
                raise ValueError("bbox: min_lat must be less than max_lat")
            if self.bbox["min_lon"] >= self.bbox["max_lon"]:
                raise ValueError("bbox: min_lon must be less than max_lon")

        if self.type not in ["csv", "parquet", None]:
            raise ValueError(f"Pick file format {self.type} is not recognized !")

        if not self.type:
            return

        for f in self.filenames:
            if not os.path.exists(f):
                raise FileNotFoundError(f"File {f} does not exist !")

            if not os.access(f, os.R_OK):
                raise PermissionError(f"{f}.")

        if self.start:
            self.start = pd.to_datetime(self.start, utc=True).to_datetime64()

        if self.end:
            self.end = pd.to_datetime(self.end, utc=True).to_datetime64()

        # Check parquet or csv file
        if self.type == "parquet":
            for f in self.filenames:
                try:
                    fastparquet.ParquetFile(f)
                except Exception as e:
                    raise ValueError(f"{f} is not parquet formated: {e}")

            # add all parquet files in the directory and subdirectories for duckdb
            self.filenames = [
                os.path.join(f, "**", "*.parquet")
                for f in self.filenames
                if os.path.isdir(f)
            ]
        else:
            # CSV columns (9) are:
            #   station_id, channel, phase_type, phase_time, phase_score,
            #   phase_evaluation, phase_method, event_id, agency
            # optional columns:
            #   month,year
            try:
                for f in self.filenames:
                    with open(f, "r", encoding="utf-8") as file:
                        first_line = file.readline().strip()
                        nbcol = len(first_line.split(","))
                        if nbcol != 9 and nbcol != 11:
                            raise ValueError(
                                f"{f} is not a csv file or some columns are missing ({nbcol}) !\n"
                                f"{first_line}"
                            )
            except Exception:
                raise

        # set min, max time from data
        ic(self.filenames, self.type)

        if self.type == "parquet":
            conn = duckdb_init_parquet(self.filenames)
        else:
            conn = duckdb_init(self.filenames, self.type)

        rqt = "SELECT MIN(phase_time), MAX(phase_time) FROM PICKS"
        results = conn.sql(rqt).fetchall()
        if not results:
            conn.close()
            raise ValueError(f"No data found in {self.filenames}")
        min, max = results[0]
        conn.close()

        # check min, max time exists
        if not min or not max:  # pragma: no cover
            raise ValueError(f"Can't find min, max time in {self.filenames}, no data ?")

        ic(min, max)

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

    default: Optional[str] = None
    hosts: Optional[Dict[str, str]] = None
    url: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.hosts:
            if self.default is not None:
                raise ValueError("FDSN default requires declared hosts !")
            # Configuration disabled: leave url unset.
            self.url = None
            return

        # check url validity for each service
        for key, value in self.hosts.items():
            if not is_valid_url(value, syntax_only=True):
                raise URLError(f"{key} URL {value} is not valid !")

        if self.default is None:
            self.url = None
        elif self.default not in self.hosts:
            raise ValueError(f"Service {self.default} not found in FDSN hosts !")
        else:
            self.url = self.hosts[self.default]

    def set_url_from_service_name(self, service: str) -> None:
        if not self.hosts:
            raise ValueError("Cannot set FDSN URL: no hosts configured !")
        if service not in self.hosts:
            raise ValueError(f"Service {service} not found in FDSN hosts !")
        self.url = self.hosts[service]

    def get_url(self) -> str:
        if self.url is None:
            raise ValueError("FDSN URL has not been configured !")
        return self.url


@dataclass
class RenameConfig:
    r"""Manage renaming rules for station codes with time-based conditions.

    example:

        rename:
            # Transformation à appliquer systématiquement avant les autres
            before:
            - r"^([^\.]+\.[^\.]+\.[^\.]+)\.[^\.]*ZNE$": r"\1.Z"

            # Transformation à appliquer systématiquement après les autres
            after:
            - r"^([^\.]+\.[^\.]+\.[^\.]+)\.[^\.]*XXZ$": r"\1.SHZ"

            # Transformations conditionnelles avec intervalles de dates
            time_windows:
            - time_window: "1970-01-01/2016-01-01"
                regex:
                - r"FR\.LBL\.00\.[H]SH.*": "FR.LBL..SH"
            - time_window: null  # Transformation sans restriction temporelle
                regex:
                - r"FR\.AGO\.00\.[SE]H[ZNE]": "FR.AGO..SH"
                - ...
    """

    before: Optional[List[Dict[str, str]]] = None  # List of regex pattern: replacement
    after: Optional[List[Dict[str, str]]] = None  # Same for transformations after
    time_windows: List[Dict[str, Union[str, Optional[List[Dict[str, str]]]]]] = None

    def __post_init__(self):
        # Ensures structure is validated if needed
        for time_window in self.time_windows:
            assert (
                "regex" in time_window
            ), "Each time_window entry must have a 'regex' key"
            if "time_window" in time_window and time_window["time_window"] is not None:
                assert (
                    "/" in time_window["time_window"]
                ), "Invalid time_window format, use 'YYYY-MM-DD/YYYY-MM-DD'"


@dataclass
class StationConfig:
    """Configure how station coordinates are obtained.

    Coordinates can be sourced from on-disk inventories, an FDSN web service, or
    CSV fallback files providing minimal metadata. The configuration also handles
    optional blacklist/rename rules as well as thresholds for filtering stations.

    Attributes:
        fetch_method:
            Strategy used to retrieve stations ("inventory", "fdsnws" or None).
        fdsnws_url:
            Legacy URL field kept for backward compatibility; prefer ``fdsnws``.
        fdsnws:
            Configuration describing the remote FDSN endpoint to query.
        inventory_files:
            Paths to StationXML files loaded when ``fetch_method`` is "inventory".
        fallback:
            CSV files containing network/station/channel coordinates used as a
            last resort or to enrich partial inventories.
        blacklist:
            Optional list of station codes that must be ignored everywhere.
        rename:
            ``RenameConfig`` rules applied to station codes after loading.
        frequency_threshold:
            Minimum acceptable station frequency used for filtering.
        inventory:
            ``Inventory`` instance populated from StationXML files when relevant.
        info_sta:
            Reference to the loaded inventory or the FDSN URL to download.
        fallback_df:
            Pandas dataframe version of concatenated fallback CSV files.

    Raises:
        ValueError: If ``fetch_method`` does not match supported strategies.
        FileNotFoundError: If declared fallback CSV files are missing.
    """

    fetch_method: str | None
    fdsnws_url: Optional[str] = None
    fdsnws: Optional[FdsnConfig] = None
    inventory_files: Optional[List[str]] = None
    fallback: Optional[List[str]] = None
    blacklist: Optional[List[str]] = None
    rename: Optional[RenameConfig] = None
    frequency_threshold: Optional[float] = None
    inventory: Optional[Inventory] = None
    info_sta: Optional[Union[Inventory, str]] = None
    fallback_df: Optional[pd.DataFrame] = None

    def __post_init__(self) -> None:
        if self.fetch_method not in ["inventory", "fdsnws", None]:
            raise ValueError("Invalid fetch_method: should be 'inventory', 'fdsnws' or None !")

        if self.fetch_method is None and not self.fallback:
            raise ValueError("fetch_method None requires at least one fallback CSV file !")

        if self.fetch_method == "inventory":
            self.inventory = Inventory()
            for f in self.inventory_files:
                logger.info(f"Reading inventory file {f}")
                self.inventory.extend(
                    read_inventory(
                        f,
                        level="channel",  # Don't load responses (memory optimization)
                    )
                )
                self.info_sta = self.inventory
        elif self.fetch_method == "fdsnws":
            if self.fdsnws is not None:
                try:
                    url = self.fdsnws.get_url()
                except ValueError as exc:
                    raise ValueError(
                        "fetch_method 'fdsnws' requires a configured FdsnConfig (hosts + default)."
                    ) from exc
                logger.debug(f"Using fdsnws {url} to get station coordinates.")
                self.info_sta = url
            elif self.fdsnws_url:
                logger.debug(
                    "Using legacy fdsnws_url configuration to get station coordinates."
                )
                self.info_sta = self.fdsnws_url
            else:
                raise ValueError(
                    "fetch_method 'fdsnws' requires either an FdsnConfig or legacy fdsnws_url."
                )
        else:
            logger.info("Station fetch disabled; relying on fallback CSV only.")
            self.info_sta = None
        

        if self.fallback:
            required_columns = {"network", "station", "latitude", "longitude"}
            optional_defaults = {
                "elevation": "",
                "location": "",
                "channel": "",
                "starttime": "",
                "endtime": "",
                "alias": "",
            }
            dtype_dict = {
                "network": "str",
                "station": "str",
                "location": "str",
                "channel": "str",
                "latitude": "float64",
                "longitude": "float64",
                "elevation": "float64",
                "starttime": "str",
                "endtime": "str",
                "alias": "str",
            }
            for f in self.fallback:
                logger.info(f"Reading fallback file {f}")
                if not os.path.exists(f):
                    raise FileNotFoundError(f"File {f} does not exist !")

                # Read only the columns present in the file to avoid dtype errors
                cols_in_file = pd.read_csv(f, nrows=0).columns.tolist()
                dtype_for_read = {k: v for k, v in dtype_dict.items() if k in cols_in_file}

                try:
                    df = pd.read_csv(f, dtype=dtype_for_read)
                except (TypeError, ValueError) as err:
                    raise ValueError(
                        "Fallback CSV parsing failed: ensure latitude/longitude/elevation "
                        "columns contain numeric values only."
                        f" File '{f}' raised: {err}"
                    ) from err
                except Exception:
                    raise

                missing_required = required_columns - set(df.columns)
                if missing_required:
                    raise ValueError(
                        "Fallback CSV is missing required columns: "
                        f"{', '.join(sorted(missing_required))}. "
                        f"Required: {', '.join(sorted(required_columns))}."
                    )

                # Add missing optional columns with default values
                for col, default in optional_defaults.items():
                    if col not in df.columns:
                        logger.info(
                            f"Fallback CSV '{f}': missing optional column '{col}', "
                            f"using default '{default or 'empty string'}'"
                        )
                        df[col] = default

                # if no elevation defined set to 0.0
                df["elevation"] = pd.to_numeric(df["elevation"], errors="coerce")
                df["elevation"] = df["elevation"].fillna(0.0)
                #  else set to empty string
                df = df.fillna("")

                # if "dateFrom" is empty, replace it with "1970-01-01"
                df["starttime"] = df["starttime"].replace("", "1970-01-01T00:00:00Z")
                # if "dateTo" is empty, replace it with "2100-01-01"
                df["endtime"] = df["endtime"].replace("", "2100-01-01T00:00:00Z")
                # Convert to UTCDateTime
                df["starttime"] = df["starttime"].apply(lambda x: UTCDateTime(x))
                df["endtime"] = df["endtime"].apply(lambda x: UTCDateTime(x))

                if self.fallback_df is None:
                    self.fallback_df = df
                else:
                    self.fallback_df = pd.concat(
                        [self.fallback_df, df], ignore_index=True
                    )


@dataclass
class TimeConfig:
    """
    Configuration class for time settings.
    """

    time_window: int  # minutes
    overlap_window: int  # seconds


@dataclass
class ClusterConfig:
    """Manages and checks parameters associated with cluster

    Raises:
        FileExistsError: if pre_computed_tt_matrix_file already exists
    """

    min_cluster_size: int
    min_station_count: int
    min_station_with_P_and_S: int
    min_station_score: float
    average_velocity: float
    min_picks_common: int
    max_search_dist: Optional[float] = 0.0
    pre_computed_tt_matrix_file: Optional[str] = None
    # Include HDBSCAN noise picks in PyOcto aggregation (as an additional cluster)
    include_noise_in_aggregation: bool = False
    # Minimum S/P pick ratio to accept a localized event (None = disabled)
    min_ps_ratio: Optional[float] = None
    # Force clusters with known event_id through pre-NLL filters even if they fail thresholds
    force_keep_catalog_events: bool = False

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
    loc_method: str
    time_path: str
    template_path: str
    default_velocity_profile: str
    velocity_profiles: List[NonLinLocVelocityProfile]
    verbose: bool
    enable_scatter: bool
    default_template_file: Optional[str] = None
    min_phase: Optional[int] = 4

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
    gap_dist_max_km: Optional[float] = None
    closest_station_dist_km: Optional[float] = None
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
    # minimum distance (degrees) to epicenter to allow pick relabeling
    # picks closer than this threshold are not relabeled (avoids polygon overlap near origin)
    min_dist_relabel_deg: Optional[float] = 0.0
    # remove all picks (including manual) with NLLoc time_weight below this threshold
    # catches clock-drift issues where NLLoc down-weights all phases from an affected station
    # set to null to disable
    min_time_weight: Optional[float] = None
    # apply P/S time residual thresholds even when use_pick_zone is True
    # picks inside a polygon but with large residuals will still be removed
    enable_residual_threshold_with_pick_zone: Optional[bool] = False
    # detect pass 2 degradation: warn if RMS_pass2 > RMS_pass1 * factor
    # set to null to disable detection entirely
    pass2_degradation_factor: Optional[float] = None
    # if True AND degradation is detected, revert to pass 1 as preferred origin
    pass2_fallback: Optional[bool] = False


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

    keep_not_existing_event: bool
    enable_quakeml_file: bool
    qml_path: str
    qml_base_filename: str
    event_flush_count: int
    #
    enable_sqlite: bool
    sqlite_db_path: str
    sqlite_db_filename: str
    sqlite_db_fullpath: Optional[str] = None
    keep_temp_db_after_merge: bool = False
    temp_db_dir: Optional[str] = None

    def __post_init__(self) -> None:
        if self.enable_quakeml_file:
            if not os.path.exists(self.qml_path) or not os.path.isdir(self.qml_path):
                try:
                    os.makedirs(self.qml_path)
                except OSError as e:
                    raise OSError(f"Can't create directory {self.qml_path}: {e}") from e

            if not os.access(self.qml_path, os.W_OK):
                raise PermissionError(f"Can't write in {self.qml_path} directory.")

        if self.enable_sqlite:
            if not os.path.exists(self.sqlite_db_path) or not os.path.isdir(
                self.sqlite_db_path
            ):
                try:
                    os.makedirs(self.sqlite_db_path)
                except OSError as e:
                    raise OSError(
                        f"Can't create directory {self.sqlite_db_path}: {e}"
                    ) from e

            if not os.access(self.sqlite_db_path, os.W_OK):
                raise PermissionError(
                    f"Can't write in {self.sqlite_db_path} directory."
                )
            self.sqlite_db_fullpath = os.path.join(
                self.sqlite_db_path, self.sqlite_db_filename
            )


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
        logger.info("Zone name: velocity profile")
        for index, row in self.polygons.iterrows():
            logger.info(f'\t{row["name"]}: {row["velocity_profile"]}')
        logger.info("")


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
    # Optional parameters to limit lat/lon ranges calculated from station coordinates
    max_lat_range: Optional[List[float]] = None  # [lat_min, lat_max]
    max_lon_range: Optional[List[float]] = None  # [lon_min, lon_max]
    # Adaptive min_pick_fraction: reduce it for clusters with multiple known event_ids
    adaptive_min_pick_fraction: bool = False
    # Floor to avoid explosively low min_pick_fraction values when adaptive mode is on
    min_pick_fraction_floor: float = 0.10
    # DL picker method_ids used to compute median proba for adaptive scaling (case-insensitive)
    dl_method_ids: List[str] = field(default_factory=lambda: ["PHASENET"])
    # Minimum S/P pick ratio to accept a PyOcto cluster (None = disabled)
    min_ps_ratio: Optional[float] = None
    # Time-slicing size in seconds for PyOcto time blocks (default 1200)
    time_slicing: float = 1200.0


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
    enable: Optional[bool] = True
    # associator and velocity_model are in current_model.keys()
    current_model: Optional[Model] = None
    travel_time_grid_filename: Optional[str] = None
    velocity_model: Optional[VelocityModel1D] = None
    delegate_dbclust: Optional[bool] = False

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

        profil_model = self.create_travel_time_grid_file(
            self.current_model.velocity_model, self.travel_time_grid_filename
        )

        # get first P and S velocity from profil_model
        # to define velocity model above surface
        vp0 = profil_model["vp"].iloc[0]
        vs0 = profil_model["vs"].iloc[0]
        logger.info(
            f"Using P velocity {vp0} and S velocity {vs0} from model {self.default_model_name}"
        )

        # Create 1D velocity model
        self.velocity_model = self.create_velocity_model(vp0=vp0, vs0=vs0)
        # self.velocity_model = self.create_velocity_model()

    def create_velocity_model(self, vp0=None, vs0=None) -> VelocityModel1D:
        tolerance = self.current_model.velocity_model.tolerance
        velocity_model = VelocityModel1D(
            path=self.travel_time_grid_filename,
            tolerance=tolerance,
            # association_cutoff_distance=None,
            # location_cutoff_distance=None,
            surface_p_velocity=vp0,
            surface_s_velocity=vs0,
        )
        return velocity_model

    def create_travel_time_grid_file(
        self, vmodel: VelocityModel, filename: str
    ) -> pd.DataFrame:
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

        return profil_model


@dataclass
class SlurmConfig:
    """Configuration for SLURM cluster execution with Parsl."""

    enabled: bool = False
    partition: str = "grant"
    account: Optional[str] = None
    nodes_per_block: int = 1
    cores_per_node: int = 32
    max_workers_per_node: int = 32
    walltime: str = "72:00:00"
    worker_init: str = "module load python; conda activate dbclust"
    scheduler_options: str = ""
    max_blocks: int = 10
    min_blocks: int = 0


@dataclass
class ParallelConfig:
    n_workers: Optional[int] = None
    partition_duration: str = "1D"
    nb_partitions: Optional[int] = None
    time_partitions: Optional[List] = None
    _temp_dir: Optional[str] = "/tmp/ray"
    executor: Optional[str] = "parsl_thread"  # parsl_thread, parsl_hte, ray, dask
    oversubscription_factor: int = 5  # workers spend ~80% waiting for NLLoc subprocess
    task_profiles_path: Optional[str] = None  # Chemin vers task_profiles.csv
    task_profiles_reference_path: Optional[str] = None  # Référence stable pour tri longest-first
    execution_summary_path: Optional[str] = None  # Chemin vers execution_summary.csv

    def __post_init__(self):
        if not self.n_workers:
            self.n_workers = os.cpu_count()

    def get_time_partitions(
        self, time_cfg: "TimeConfig", pick_cfg: "PickConfig"
    ) -> List:
        # Convertir proprement en Timestamp et arrondir
        start = pd.to_datetime(pick_cfg.start).replace(second=0, microsecond=0)
        original_end = pd.to_datetime(pick_cfg.end)

        end = original_end.replace(second=0, microsecond=0)
        if original_end > end:
            end += pd.Timedelta(minutes=1)

        duration = pd.Timedelta(self.partition_duration)
        total_duration = end - start
        nb_full_partitions = math.floor(total_duration / duration)
        remainder = total_duration % duration

        if remainder > pd.Timedelta(0):
            nb_full_partitions = nb_full_partitions + 1
        end = start + nb_full_partitions * duration
        logger.info(
            f"start: {start}, end: {end}, duration: {duration}, nb_partitions: {nb_full_partitions}"
        )

        time_divisions = pd.date_range(
            start=start, end=end, freq=duration, inclusive="left"
        ).to_list()

        # adjusted_time_divisions = [
        #     (s, s + duration + pd.Timedelta(seconds=time_cfg.overlap_window))
        #     for s in time_divisions
        # ]
        # self.nb_partitions = len(adjusted_time_divisions)
        # return adjusted_time_divisions

        adjusted_time_divisions = [(s, s + duration) for s in time_divisions]
        self.nb_partitions = len(time_divisions)

        return adjusted_time_divisions


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
    slurm: Optional[SlurmConfig] = None

    def __init__(self, filename, config_type="std") -> None:
        # config_type can be "std" or "reloc"
        # reloc is used for relocation only
        # std is used for standard processing
        self.reloc_no_required_keys = [
            "parallel",
            "pick",
            "time",
            "catalog",
            "station",
        ]
        # Optional config sections (not required in YAML)
        self.optional_keys = ["slurm"]
        # Runtime attribute set by runner, not from YAML
        self.log_level = logging.INFO

        self.filename = filename
        logger.info(filename)
        self.config_type = config_type

        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"File {self.filename} does not exist !")
        self.yaml_data = read_config(self.filename)

        for key, data_class in self.__annotations__.items():
            if self.config_type == "reloc" and key in self.reloc_no_required_keys:
                # skip some config not needed for reloc
                logger.warning(f"Ignoring section '{key}' in yaml file !")
                continue
            elif key in self.optional_keys:
                # Optional sections: use defaults if not in YAML
                if key in self.yaml_data.keys():
                    # Extract inner type from Optional[X] -> X
                    inner_type = (
                        data_class.__args__[0]
                        if hasattr(data_class, "__args__")
                        else data_class
                    )
                    setattr(
                        self,
                        key,
                        from_dict(data_class=inner_type, data=self.yaml_data[key]),
                    )
                else:
                    # Use default values from dataclass
                    inner_type = (
                        data_class.__args__[0]
                        if hasattr(data_class, "__args__")
                        else data_class
                    )
                    setattr(self, key, inner_type())
            else:
                if key not in self.yaml_data.keys():
                    raise ValueError(f"Missing section '{key}' in yaml file !")
                setattr(
                    self,
                    key,
                    from_dict(data_class=data_class, data=self.yaml_data[key]),
                )

        # NLL will discard any location with number of phase < min_phase
        # take into account cluster parameters to set it accordingly
        # use -1 to not set a limit
        # self.nll.min_phase = (
        #     self.cluster.min_station_count + self.cluster.min_station_with_P_and_S
        # )

        if config_type != "reloc":
            # parallel
            self.parallel.time_partitions = self.parallel.get_time_partitions(
                self.time, self.pick
            )
            assert len(self.parallel.time_partitions)

            # Apply geographic filtering on stations if bbox is defined
            if self.pick.bbox:
                from dbclust.db import filter_inventory_by_bbox
                from dbclust.db import filter_stations_by_bbox

                if self.station.inventory is not None:
                    self.station.inventory = filter_inventory_by_bbox(
                        self.station.inventory,
                        self.pick.bbox,
                    )
                    self.station.info_sta = self.station.inventory

                if self.station.fallback_df is not None:
                    self.station.fallback_df = filter_stations_by_bbox(
                        self.station.fallback_df,
                        self.pick.bbox,
                    )

        # Finalize zones
        self.zones.load_zones(self.nll)

        # Set default velocity model
        self.quakeml.model_id = self.nll.default_velocity_profile
        # ic(self.quakeml)

    def show(self):
        # debug
        for key, value in self.__annotations__.items():
            if self.config_type == "reloc" and key in self.reloc_no_required_keys:
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
            with urlopen(url, timeout=10):
                pass
            return True
    except (URLError, TimeoutError):
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
