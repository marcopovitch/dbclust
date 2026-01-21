#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import glob
import json
import logging
import os
import re
import shlex
import sys
import tempfile
import traceback
import urllib.parse
import urllib.request
import uuid
import warnings
from collections import defaultdict
from itertools import combinations
from math import fabs
from math import isclose
from typing import List
from typing import Tuple
from typing import Union

import dateparser
import geopandas as gpd
import numpy as np
import pandas as pd
from jinja2 import Template
from obspy import Catalog
from obspy import read_events
from obspy.core import UTCDateTime
from obspy.core.event import Arrival
from obspy.core.event import Comment
from obspy.core.event import CreationInfo
from obspy.core.event import Event
from obspy.core.event import Origin
from obspy.core.event import OriginQuality
from obspy.core.event import Pick
from obspy.core.event import ResourceIdentifier
from obspy.core.event import WaveformStreamID
from obspy.geodetics import gps2dist_azimuth
from obspy.geodetics import kilometer2degrees
from prettytable import PrettyTable
from shapely.geometry import Point

from dbclust.config import Zone
from dbclust.config import Zones
from dbclust.gap import compute_azimuthal_gap
from dbclust.gap import compute_gap
from dbclust.gap import compute_secondary_azimuthal_gap
from dbclust.gap import get_arrival_with_distance_gap_greater_than
from dbclust.localization_quality import classify_event, classify_event_michele_mod2    
from dbclust.plot import plot_arrival_time
from dbclust.quakeml import deduplicate_picks
from dbclust.relabel import get_best_polygon_for_point
from dbclust.relabel import relabel_phase_and_comment_arrival

# Disable warnings from obspy
# UserWarning: Setting attribute ... which is not a default attribute
warnings.filterwarnings("ignore", category=UserWarning, module="obspy")

# default logger (uses hierarchical name for selective level control)
logger = logging.getLogger("dbclust.localization")


# Define the preferred phase order
phase_order = ["Pg", "Sg", "Pn", "Sn", "P", "S"]

# set time_weight tolerance
time_weight_tolerance = 0.01


def safe_subprocess_run(args, *, env=None, cwd=None, text=True):
    """
    Simplified replacement for subprocess.run(..., stdout=PIPE, stderr=STDOUT, text=True)
    using posix_spawnp (safe for macOS).
    """
    if isinstance(args, str):
        args = shlex.split(args)

    env = env or os.environ.copy()

    # Temporary file to capture stdout/stderr
    fd, tmpfile = tempfile.mkstemp()
    os.close(fd)
    fd_out = os.open(tmpfile, os.O_WRONLY | os.O_TRUNC)

    file_actions = [
        (os.POSIX_SPAWN_DUP2, fd_out, 1),  # stdout
        (os.POSIX_SPAWN_DUP2, fd_out, 2),  # stderr
    ]

    # Temporarily change working directory if requested
    old_cwd = None
    if cwd:
        old_cwd = os.getcwd()
        os.chdir(cwd)

    try:
        pid = os.posix_spawnp(
            args[0],
            args,
            env,
            file_actions=file_actions,
        )
    finally:
        if cwd and old_cwd:
            os.chdir(old_cwd)
        os.close(fd_out)

    # Wait for process to finish
    _, status = os.waitpid(pid, 0)
    returncode = os.waitstatus_to_exitcode(status)

    # Read output
    with open(tmpfile, "r", errors="replace") as f:
        output = f.read()
    os.unlink(tmpfile)

    # Result object similar to subprocess
    class Result:
        pass

    result = Result()
    result.returncode = returncode
    result.stdout = output if text else output.encode()

    return result


class LocalizationError(Exception):
    """Raised when NonLinLoc localization fails."""

    def __init__(self, txt: str):
        super().__init__(f"Localization failed: {txt}")
        self.txt = txt


def sort_by_phase(arrival: Arrival) -> int:
    """
    Sorts an Arrival object by its phase.
    Parameters:
        arrival (Arrival)
    Returns:
        int: index of the phase in the predefined phase_order list. If not found returns the length of phase_order.
    """
    phase = arrival.phase  # Assuming the phase is stored in arrival.phase
    return phase_order.index(phase) if phase in phase_order else len(phase_order)


def sort_by_cluster_file(filename: str) -> float:
    match = re.search(r"cluster-(\d+)\.obs", filename)
    if match:
        # return the number to be used by sorted()
        return int(match.group(1))
    else:
        return float("inf")


class NllLoc(object):
    def __init__(
        self,
        nll_bin,
        scat2latlon_bin,
        nll_times_path,
        nll_template=None,
        nll_obs_file=None,
        nll_min_phase=4,
        nll_verbose=False,
        nll_default_template=None,  # default template to use if no preloc found
        loc_method="EDT_OT_WT_ML",
        tmpdir="/tmp",
        min_station_score=5.5,
        min_station_with_P_and_S=0,
        double_pass=False,
        force_uncertainty=False,
        P_uncertainty=0.1,
        S_uncertainty=0.2,
        gap_dist_max_km=100,
        dist_km_cutoff=None,
        use_deactivated_arrivals=False,
        keep_manual_picks=False,
        P_time_residual_threshold=None,
        S_time_residual_threshold=None,
        quakeml_settings=None,
        keep_scat=False,
        scat_file=None,
        zones: Zones = None,  # zones (polygons) delimitation to keep picks
        force_zone_name: str = None,  # force zone to use
        min_score_threshold_pick_zone=0.5,  # minimum score to relabel pick in zone
        use_pick_zone: bool = True,  # use pick zones
        enable_cleanup_pick_zone: bool = True,  # clean up pick outside of zone
        enable_relabel_pick_zone: bool = False,  # relabel pick within zone
        keep_not_existing_event: bool = False,  # keep "not existing" event, or not
    ):
        # define locator
        self.nll_bin = nll_bin
        self.scat2latlon_bin = scat2latlon_bin
        self.nll_time_path = nll_times_path
        self.nll_template = nll_template
        self.nll_obs_file = nll_obs_file  # obs file to localize
        self.nll_min_phase = nll_min_phase
        self.nll_verbose = nll_verbose
        self.nll_default_template = nll_default_template
        self.loc_method = loc_method
        self.tmpdir = tmpdir
        self.min_station_score = min_station_score
        self.min_station_with_P_and_S = min_station_with_P_and_S
        self.double_pass = double_pass
        self.force_uncertainty = force_uncertainty
        self.P_uncertainty = P_uncertainty
        self.S_uncertainty = S_uncertainty
        self.gap_dist_max_km = gap_dist_max_km
        self.dist_km_cutoff = dist_km_cutoff
        self.use_deactivated_arrivals = use_deactivated_arrivals
        self.keep_manual_picks = keep_manual_picks
        self.P_time_residual_threshold = P_time_residual_threshold
        self.S_time_residual_threshold = S_time_residual_threshold
        self.quakeml_settings = quakeml_settings
        self.keep_scat = keep_scat
        self.scat_file = scat_file
        self.zones = zones
        self.force_zone_name = force_zone_name
        self.min_score_threshold_pick_zone = min_score_threshold_pick_zone
        self.use_pick_zone = use_pick_zone
        self.enable_cleanup_pick_zone = enable_cleanup_pick_zone
        self.enable_relabel_pick_zone = enable_relabel_pick_zone
        self.keep_not_existing_event = keep_not_existing_event

        # keep track of cluster affiliation
        self.event_cluster_mapping = {}

        # localization only if is nll_obs_file provided at init level
        if self.nll_obs_file:
            try:
                self.catalog = self.nll_localisation(
                    nll_obs_file, double_pass=self.double_pass
                )
            except LocalizationError as e:
                logger.error(f"{e} - Check your input data or parameters.")
                self.catalog = Catalog()
            except Exception as e:
                logger.error(f"Unexpected error during localization: {e}")
                self.catalog = Catalog()
        else:
            self.catalog = Catalog()
        self.nb_events = len(self.catalog)

    @staticmethod
    def check_stations_with_P_and_S(
        event: Event, origin: Origin, min_count: int
    ) -> int:
        """
        Ensures that the number of stations with both P and S phases (count)
        is greater than or equal to the threshold (min_count).

        Returns count
        """
        count = {}
        for arrival in origin.arrivals:
            if hasattr(arrival, "time_weight") and isclose(
                arrival.time_weight, 0, abs_tol=time_weight_tolerance
            ):
                continue
            pick = get_pick_from_arrival(event, arrival)
            if pick is None:
                continue
            wfid = pick.waveform_id
            station_name = f"{wfid.network_code}.{wfid.station_code}"
            phase_name = arrival.phase

            if station_name in count.keys():
                count[station_name].append(phase_name)
            else:
                count[station_name] = [phase_name]

        count = [len(count[k]) for k in count.keys()]
        return np.array([np.count_nonzero(x >= min_count) for x in count]).sum()

    @staticmethod
    def get_origin_station_score(event: Event, origin: Origin) -> float:
        arrivals = origin.arrivals
        if not arrivals:
            return 0.0

        station_phases = defaultdict(set)

        for arrival in arrivals:
            if arrival.time_weight is None or arrival.time_weight == 0:
                continue
            pick_id = arrival.pick_id
            pick = next((p for p in event.picks if p.resource_id == pick_id), None)
            if pick is None or pick.waveform_id is None:
                continue
            net = pick.waveform_id.network_code
            sta = pick.waveform_id.station_code
            station_code = f"{net}.{sta}"

            phase = arrival.phase.lower()
            if phase.startswith("p"):
                station_phases[station_code].add("P")
            elif phase.startswith("s"):
                station_phases[station_code].add("S")

        score = 0.0
        for phases in station_phases.values():
            if "P" in phases and "S" in phases:
                score += 2.0
            elif "P" in phases:
                score += 1.0
            elif "S" in phases:
                score += 0.5

        return score

    def _get_zone_and_template(
        self, lat: float, lon: float, vel_file: str = None
    ) -> Tuple:
        """
        Detect zone and return appropriate template/model_id.

        Args:
            lat: origin latitude
            lon: origin longitude
            vel_file: path to .vel file (prelocalization), if exists

        Returns:
            Tuple (zone, nll_template, model_id)
        """
        zone = None
        nll_template = self.nll_template
        model_id = (
            self.quakeml_settings.get("model_id") if self.quakeml_settings else None
        )

        if not self.zones:
            return zone, nll_template, model_id

        logger.info(f"Detecting zone for lat: {lat}, lon: {lon}")

        # Zone detection
        if self.force_zone_name:
            zone = self.zones.get_zone_from_name(self.force_zone_name)
        else:
            zone, _ = self.zones.find_zone(lat, lon)

        # If no preloc and zone found, use the zone's template
        # Note: zone can be a pd.Series (row) or empty GeoDataFrame
        zone_is_valid = zone is not None and len(zone) > 0
        if (
            zone_is_valid
            and (vel_file is None or not os.path.exists(vel_file))
            and zone["template"]
        ):
            nll_template = zone["template"]
            model_id = zone["velocity_profile"]
            zone_name = zone["name"]
            logger.info(
                f"Zone '{zone_name}' found. "
                f"Using template: {nll_template}, model: {model_id}"
            )

        logger.info(f"Using template: {nll_template}, model: {model_id}")

        return zone, nll_template, model_id

    def reloc_event(self, event: Event) -> Catalog:
        """Event re-localization using a locator.

        Args:
            event (Event): The seismic event to be re-localized.

        Returns:
            Catalog: A catalog containing the re-localized event.

        Raises:
            LocalizationError: If the localization fails.
            Exception: If there is an unexpected error during localization.
        """
        myevent = copy.deepcopy(event)
        show_event(myevent, "****", header=True)
        orig = myevent.preferred_origin()
        mypicks = []

        for arrival in orig.arrivals:
            # pick = arrival.pick_id.get_referred_object()
            pick = get_pick_from_arrival(myevent, arrival)
            if pick is None:
                logger.warning("Pick not found for arrival %s", arrival)
                continue

            # Ensure pick phase_hint is the same as arrival phase
            # in order to avoid phase mismatch in the nll_obs input file
            pick.phase_hint = arrival.phase

            if self.force_uncertainty:
                phase_upper = arrival.phase.upper()
                if "P" in phase_upper:
                    pick.time_errors.uncertainty = self.P_uncertainty
                elif "S" in phase_upper:
                    pick.time_errors.uncertainty = self.S_uncertainty

            # Remove picks associated with deactivated arrivals unless explicitly allowed.
            if not self.use_deactivated_arrivals and isclose(
                arrival.time_weight, 0, abs_tol=time_weight_tolerance
            ):
                continue

            # Remove arrivals from stations if their distance exceeds the cutoff.
            if self.dist_km_cutoff is not None and arrival.distance > (
                self.dist_km_cutoff / 111.0
            ):
                continue

            mypicks.append(pick)

        myevent.picks = mypicks

        self.nll_obs_file = os.path.join(self.tmpdir, "nll_obs.txt")
        logger.debug(
            f"Writing NLLoc observation file to {self.nll_obs_file} in {self.tmpdir} directory."
        )

        # NLLoc format only requires pick information, not arrivals.
        if len(myevent.picks) == 0:
            logger.warning("No picks found for localization.")
            raise LocalizationError("no picks found.")

        myevent.write(self.nll_obs_file, format="NLLOC_OBS")

        try:
            cat = self.nll_localisation(picks=myevent.picks)
        except LocalizationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during localization: {e}")
            traceback.print_exc()
            raise

        # Add the previous event or origin back to this event.
        if cat:
            new_loc = cat.events[0]
            new_loc.origins.append(orig)
            new_loc.picks.extend(event.picks)
        else:
            raise LocalizationError(f"using {self.loc_method} method.")

        return cat

    def _load_nll_event(
        self, nll_output_path: str, tmp_path: str, picks: List[Pick], stdout_text: str
    ) -> Catalog | None:
        """Load NonLinLoc results and handle diagnostics."""
        if not os.path.exists(nll_output_path):
            logger.warning(
                f"Localization failed: NLL output file not found: {nll_output_path}"
            )
            logger.warning(f"NLL stdout:\n{stdout_text}")
            try:
                files = os.listdir(tmp_path)
                logger.info(f"Files in {tmp_path}: {files}")
            except Exception:
                logger.info("Unable to list temporary directory contents", exc_info=True)
            return None

        try:
            return read_events(nll_output_path, picks=picks)
        except Exception as exc:
            logger.warning(f"Localization failed: unable to read NLL output ({exc})")
            return None

    def nll_localisation(
        self,
        nll_obs_file: str = None,
        picks: List[Pick] = None,
        double_pass: bool = None,
        pass_count: int = 0,
        force_model_id: str = None,
        force_template: str = None,
        force_loc_method: str = None,
    ):
        """
        Perform NonLinLoc localization for seismic events.

        Parameters:
        -----------
        nll_obs_file : str, optional
            Path to the NLL observation file. If not provided, uses the instance's default.
        picks : List[Pick], optional
            List of Pick objects to be used in localization.
        double_pass : bool, optional
            If True, perform a double pass localization.
        pass_count : int, optional
            Counter for the number of localization passes.
        force_model_id : str, optional
            Force the use of a specific model ID.
        force_template : str, optional
            Force the use of a specific template.

        Returns:
        --------
        Catalog
            A Catalog object containing the localized event(s).

        Raises:
        -------
        LocalizationError
            If the localization fails using the specified method.
        Exception
            If there is an error in generating the NLL configuration file or running the NLL binary.
        """

        # ic(nll_obs_file, double_pass, pass_count, force_model_id, force_template)

        if not nll_obs_file:
            nll_obs_file = self.nll_obs_file

        if not nll_obs_file:
            logger.error("No NLL_OBS file given !")
            return Catalog()

        if double_pass is not None:
            # force double pass
            self.double_pass = double_pass
        # else: use the value defined in locator

        # use default template
        if not force_template:
            nll_template = self.nll_template
        else:
            # defined by double pass
            nll_template = force_template

        # defined model_id
        if not force_model_id:
            if (
                self.quakeml_settings
                and "model_id" in self.quakeml_settings
                and self.quakeml_settings["model_id"]
            ):
                model_id = self.quakeml_settings["model_id"]
            else:
                model_id = os.path.basename(nll_template)
        else:
            # defined by double pass
            model_id = force_model_id

        # check if .vel file is available to force localization
        # using this model/template
        # create an origin associated to this prelocalization
        vel_file = os.path.splitext(nll_obs_file)[0] + ".vel"
        preloc_origin = None
        preloc_picks_list = []
        # get velocity model to use thanks to preliminary location
        if os.path.exists(vel_file):
            with open(vel_file, encoding="utf-8") as vel:
                model_id = vel.readline().strip()
                nll_template = vel.readline().strip()

            logger.info(
                f"Preloc forces localization to use model_id: {model_id}, "
                f"template: {nll_template}."
            )

        # ic(picks)

        if pass_count == 0:
            # get info to create a full Origin for preliminary location
            # (only on the first location iteration)
            if os.path.exists(vel_file):
                logger.info("Creating a preliminary location from pyocto.")
                picks_file = os.path.splitext(nll_obs_file)[0] + "-picks.csv"
                sta_file = os.path.splitext(nll_obs_file)[0] + "-sta.csv"
                # pyocto has generated a preloc
                preloc_origin, preloc_picks_list = make_preloc_origin(
                    vel_file, picks_file, sta_file, self.quakeml_settings
                )
                # logger.info("Preloc origin created: %s", preloc_origin, exc_info=True)
            else:
                # pyocto has not generated a preloc (or used to relocate the event)
                # due to clusters obtained from dbscan only.
                # Use only the default template and velocity model
                if not self.nll_default_template:
                    if self.nll_template:
                        self.nll_default_template = self.nll_template
                    else:
                        logger.error(
                            "No preloc file found and no default nll template provided !"
                        )
                        return Catalog()
                nll_template = self.nll_default_template
                logger.info(
                    f"No preloc file found. Using default nll template and model {os.path.basename(nll_template)}"
                )

        logger.debug(f"Localization of {nll_obs_file} using {nll_template} template.")
        nll_obs_file_basename = os.path.basename(nll_obs_file)

        tmp_path = tempfile.mkdtemp(dir=self.tmpdir)
        logger.debug(f"Temporary directory created: {tmp_path}")

        conf_file = os.path.join(tmp_path, f"{nll_obs_file_basename}.conf")

        # path + root filename
        output = os.path.join(tmp_path, nll_obs_file_basename)

        # Values to be substituted in the template
        tags = {
            "OBSFILE": nll_obs_file,
            "NLL_TIME_PATH": self.nll_time_path,
            "OUTPUT": output,
            "NLL_MIN_PHASE": self.nll_min_phase,
            # Apply GAU_ANALYTIC only for the first pass if double_pass is enabled (to speed up the process)
            # Warning: GAU_ANALYTIC do not always work as expected
            # "LOC_METHOD": "GAU_ANALYTIC" if (double_pass and pass_count == 0) else self.loc_method,
            "LOC_METHOD": (
                self.loc_method if force_loc_method is None else force_loc_method
            ),
        }

        # Generate NLL configuration file
        try:
            self.replace(nll_template, conf_file, tags)
        except Exception:
            logger.error(
                f"Failed to generate NLL config: template={nll_template}, conf={conf_file}, tags={tags}"
            )
            raise

        ####################
        # NLL Localization #
        ####################

        cmde = f"{self.nll_bin} {conf_file}"
        logger.debug(cmde)

        result = safe_subprocess_run(
            shlex.split(cmde),
            env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )
        if result.returncode != 0:
            logger.error(
                f"!!! Something went wrong using: {cmde}, "
                f"returned code is {result.returncode}\n"
                f"{result.stdout}"
            )
            for p in picks if picks else []:
                logger.error(p)
            return Catalog()

        # Initialize variables that may be set conditionally in the loop
        scatter_volume = None
        expect_lat = None
        expect_lon = None
        expect_depth = None

        # check from stdout if there is any missing station grid file
        for line in result.stdout.splitlines():
            if "WARNING: cannot open grid buffer file" in line:
                logger.error(line)
            elif "WARNING: too few observations to locate" in line:
                logger.error(line)
                return Catalog()
            elif any(k in line for k in ("ABORTED", "IGNORED", "REJECTED")):
                # check if location was rejected
                why = (
                    " ".join(line.split()[3:]).replace('"', "").replace("WARNING: ", "")
                )
                logger.warning(f"Localization was ABORTED|IGNORED|REJECTED: {why}")
                if self.nll_verbose:
                    print(result.stdout)
                return Catalog()
            elif any(
                k in line
                for k in (
                    "ERROR: reading x-sheet grid file",
                    "ERROR: reading lower arrival travel time sheet",
                )
            ):
                # ERROR is not fatal here as the location can be done
                logger.warning(f"This is not fatal: {line}")
            elif "x-sheet" in line:
                l = line.split()
                station_name = l[2] if len(l) > 2 else "unknown"
                logger.warning(
                    f"Station {station_name} outside velocity bounding box coordinates: localization aborted by NonLinLoc !"
                )
                if self.nll_verbose:
                    print(result.stdout)
                return Catalog()
            elif "ERROR: calc_maximum_likelihood_ot:" in line:
                # localization failed. It happens when using EDT_OT_WT
                # raise an exception to try relocation with another method
                loc_method_used = (
                    self.loc_method if force_loc_method is None else force_loc_method
                )
                raise LocalizationError(f"using {loc_method_used} method.")
            elif "ERROR" in line:
                logger.error(line)
                if self.nll_verbose:
                    print(result.stdout)
                return Catalog()
            elif "scatter_volume" in line:
                l = line.split("scatter_volume")
                if len(l) > 1:
                    scatter_volume = l[1].strip()
                else:
                    scatter_volume = None
            elif "ExpectLat" in line:
                # get expectation hypocenter
                l = line.split()
                if len(l) > 6:
                    expect_lat = float(l[2])
                    expect_lon = float(l[4])
                    expect_depth = float(l[6])

        if self.nll_verbose:
            print(result.stdout)

        # Read results
        nll_output = os.path.join(tmp_path, "last.hyp")
        cat = self._load_nll_event(nll_output, tmp_path, picks, result.stdout)
        if cat is None:
            return Catalog()

        ####################
        # handle scat file #
        ####################
        if self.keep_scat:
            # scat2latlon <decim_factor> <output_dir> <hyp_file_list>
            decim_factor = 10
            cmde = f"{self.scat2latlon_bin} {decim_factor} {tmp_path} {tmp_path}/last"
            logger.debug(cmde)
            try:
                result = safe_subprocess_run(shlex.split(cmde))
            except Exception as e:
                logger.error(e)

            self.scat_file = os.path.join(tmp_path, "last.hyp.scat.xyz")
            logger.debug("nll scat file is %s", self.scat_file)

        # there is always only one event in the catalog
        # fixme: use resource_id to forge *better* eventid and originid
        e = cat.events[0]
        o = e.preferred_origin()

        # Count arrivals returned by NonLinLoc before any cleanup
        total_arrivals = len(o.arrivals)
        zero_weight_arrivals = sum(
            1
            for a in o.arrivals
            if hasattr(a, "time_weight")
            and isclose(a.time_weight, 0, abs_tol=time_weight_tolerance)
        )
        if zero_weight_arrivals > 0:
            logger.info(
                f"NonLinLoc returned {total_arrivals} arrivals "
                f"({zero_weight_arrivals} with time_weight=0)"
            )

        # Check if any picks were lost by NonLinLoc
        if picks and len(picks) != total_arrivals:
            logger.warning(
                f"NonLinLoc returned {total_arrivals} arrivals but {len(picks)} picks were sent. "
                f"This is usually caused by co-located stations (same NET.STA code but different channels) "
                f"where NonLinLoc merges picks into a single arrival."
            )
            # Find missing picks
            arrival_pick_ids = {a.pick_id.id for a in o.arrivals if a.pick_id}
            for pick in picks:
                if pick.resource_id.id not in arrival_pick_ids:
                    logger.warning(
                        f"  Missing arrival for pick: {pick.waveform_id.get_seed_string()} "
                        f"{pick.phase_hint} {pick.time}"
                    )

        o.quality.used_station_count = self.get_used_station_count(e, o)
        o.quality.used_phase_count = self.get_used_phase_count(e, o)

        # set time_weight to 0 for arrival if time_weight < time_weight_tolerance
        # to avoid any issue with seiscomp
        for arrival in o.arrivals:
            if hasattr(arrival, "time_weight") and isclose(
                arrival.time_weight, 0, abs_tol=time_weight_tolerance
            ):
                arrival.time_weight = 0

        # check for nan value in uncertainty
        if "nan" in [
            str(o.latitude_errors.uncertainty),
            str(o.longitude_errors.uncertainty),
            str(o.depth_errors.uncertainty),
        ]:
            logger.warning("Found NaN value in uncertainty. Ignoring event !")
            return Catalog()

        if not self.quakeml_settings:
            o.creation_info.agency_id = "MyAgencyId"
            o.creation_info.author = "DBClust"
            o.evaluation_mode = "automatic"
            o.method_id = "NonLinLoc"
            o.earth_model_id = model_id
        else:
            o.creation_info.agency_id = self.quakeml_settings["agency_id"]
            o.creation_info.author = self.quakeml_settings["author"]
            o.evaluation_mode = self.quakeml_settings["evaluation_mode"]
            o.method_id = self.quakeml_settings["method_id"]
            o.earth_model_id = model_id
        # to keep track of different origins
        o.creation_info.version = pass_count + 1

        # store into comment scatter volume
        if scatter_volume is not None:
            o.comments.append(Comment(text='{"scatter_volume": %s}' % (scatter_volume)))

        # store into comment expectation hypocenter
        if (
            expect_lat is not None
            and expect_lon is not None
            and expect_depth is not None
        ):
            o.comments.append(
                Comment(
                    text='{"expectation": {"latitude": %s, "longitude": %s, "depth": %s}}'
                    % (expect_lat, expect_lon, expect_depth)
                )
            )

        if self.force_uncertainty:
            for pick in e.picks:
                if "P" in pick.phase_hint.upper():
                    pick.time_errors.uncertainty = self.P_uncertainty
                elif "S" in pick.phase_hint.upper():
                    pick.time_errors.uncertainty = self.S_uncertainty

        # Log successful first pass localization
        logger.info(
            f"Pass {pass_count + 1} localization: "
            f"lat={o.latitude:.4f}, lon={o.longitude:.4f}, depth={o.depth/1000:.1f}km, "
            f"RMS={o.quality.standard_error:.3f}, phases={o.quality.used_phase_count}, "
            f"model={model_id}"
        )

        # try a relocation
        if self.double_pass and pass_count == 0:
            logger.info("Starting double pass relocation.")
            cat2 = cat.copy()
            event2 = cat2.events[0]
            # event2 = deduplicate_picks(event2)

            # unset arrival with gap in distance > dist_max
            event2 = self.unset_arrival_gap_dist_km(event2, self.gap_dist_max_km)

            # Detect zone based on first pass location (for pick cleanup)
            zone, zone_template, zone_model_id = self._get_zone_and_template(
                o.latitude, o.longitude
            )
            # Update template/model_id only if:
            # - No preloc (.vel doesn't exist)
            # - No forced zone (self.force_zone_name is None) - if zone is forced, template was already set
            # - No forced template from caller
            if (
                not os.path.exists(vel_file)
                and not self.force_zone_name
                and not force_template
                and zone_template
            ):
                # No preloc and no forced zone: use detected zone's template for the second pass
                nll_template = zone_template
                model_id = zone_model_id
                logger.info(
                    f"Second pass: using zone '{zone['name']}' template: {nll_template}, model: {model_id}"
                )

            # Clean up picks based on zone polygons
            if self.use_pick_zone and self.zones and self.enable_cleanup_pick_zone:
                if zone is not None and len(zone) > 0 and len(zone.picks_delimiter):
                    event2, relabel_dict = self.cleanup_picks_and_relabel_picks(
                        event2, zone, eval_threshold=self.min_score_threshold_pick_zone
                    )
                else:
                    event2 = self.cleanup_pick_phase(event2)
                    relabel_dict = {}
            else:
                event2 = self.cleanup_pick_phase(event2)
                relabel_dict = {}

            if len(event2.picks):
                new_nll_obs_file = nll_obs_file + ".2nd_pass"
                logger.info(
                    f"Writing {len(event2.picks)} picks to NLLOC_OBS file for second pass"
                )
                cat2.write(new_nll_obs_file, format="NLLOC_OBS")
                # Verify how many lines were actually written
                with open(new_nll_obs_file, "r") as f:
                    nll_obs_lines = sum(1 for line in f if line.strip() and not line.startswith("#"))
                if nll_obs_lines != len(event2.picks):
                    logger.warning(
                        f"NLLOC_OBS file has {nll_obs_lines} entries but {len(event2.picks)} picks were expected"
                    )
                loc_method_used = (
                    self.loc_method if force_loc_method is None else force_loc_method
                )
                try:
                    cat2 = self.nll_localisation(
                        new_nll_obs_file,
                        picks=event2.picks,
                        double_pass=self.double_pass,
                        pass_count=1,
                        force_model_id=model_id,
                        force_template=nll_template,
                        force_loc_method=loc_method_used,
                    )
                except LocalizationError as ex:
                    logger.warning(f"Localization failed in second pass: {ex}")
                    # retry a relocation with forced LOC_METHOD = "GAU_ANALYTIC"
                    if loc_method_used != "GAU_ANALYTIC":
                        logger.info("Retrying localization with GAU_ANALYTIC as last resort")
                        try:
                            cat2 = self.nll_localisation(
                                new_nll_obs_file,
                                picks=event2.picks,
                                double_pass=self.double_pass,
                                pass_count=1,
                                force_model_id=model_id,
                                force_template=nll_template,
                                force_loc_method="GAU_ANALYTIC",
                            )
                        except Exception as ex2:
                            logger.warning(f"Localization failed with GAU_ANALYTIC fallback: {ex2}")
                            cat2 = None
                    else:
                        cat2 = None
                except Exception as ex:
                    logger.warning(f"Localization failed in second pass: unexpected error ({ex})")
                    cat2 = None
            else:
                logger.warning("Localization failed: no picks remaining for second pass")
                cat2 = None

            if cat2:
                # there is always only one event in the catalog
                event2 = cat2.events[0]
                orig2 = event2.preferred_origin()

                # Synchronize current event phase's comments with relabel_dict info
                # 1. add relabel phase according to relabel_dict
                # 2. deactivate arrival if needed
                if self.enable_cleanup_pick_zone and relabel_dict:
                    for arrival in orig2.arrivals:
                        pick = next(
                            (
                                p
                                for p in event2.picks
                                if p.resource_id == arrival.pick_id
                            ),
                            None,
                        )
                        if pick is None:
                            continue
                        key = f"{pick.waveform_id.get_seed_string()}-{arrival.phase}-{pick.time}"
                        if key in relabel_dict.keys():
                            arrival.comments.append(relabel_dict[key])
                            c = relabel_dict[key].text
                            info = json.loads(c)
                            if "removed" in info["relabel"]["action"]:
                                # deactivate arrival
                                arrival.time_weight = 0

                # add this new origin to catalog and set it as preferred
                # Remap arrivals in orig2 to reference picks in e.picks (not event2.picks)
                # because event2 has new pick IDs generated by obspy
                self._remap_arrivals_to_existing_picks(orig2, event2, e)
                e.origins.append(orig2)
                e.preferred_origin_id = orig2.resource_id
            else:
                # can't relocate: set it to "not existing"
                logger.warning("Localization failed: second pass relocation unsuccessful")
                e.event_type = "not existing"
                if not self.keep_not_existing_event:
                    # do not keep "not existing" event
                    return Catalog()

        else:
            # pass_count > 0
            pass

        # if there is only one origin, set it to the preferred
        if len(e.origins) == 1:
            e.preferred_origin_id = e.origins[0].resource_id

        # add preloc origin to event at the end
        if preloc_origin and self.double_pass and pass_count == 0:
            # Remap preloc arrivals to existing picks in e, or add missing picks
            self._remap_arrivals_to_picks(preloc_origin, preloc_picks_list, e)
            e.origins.append(preloc_origin)

        e = deduplicate_picks(e)
        return cat

    def get_catalog_from_results(self, cat_results: List[Catalog]) -> Catalog:
        """Compute attributes and filter events from catalogs"""
        final_catalog = Catalog()
        for cat in cat_results:
            if not cat or not cat.events:
                logger.debug("Empty catalog or missing events, skipping.")
                continue

            # Each catalog is expected to have exactly one event
            e = cat.events[0]
            o = e.preferred_origin()

            # Compute quality attributes
            o.quality.used_station_count = self.get_used_station_count(e, o)
            o.quality.used_phase_count = self.get_used_phase_count(e, o)

            station_score = self.get_origin_station_score(e, o)
            logger.info(
                f"Evaluating event: station score = {station_score}, "
                f"({o.quality.used_station_count} stations, {o.quality.used_phase_count} phases)"
            )

            if self.min_station_score is not None:
                if station_score < self.min_station_score:
                    # station score not enough
                    logger.info(
                        f"Rejected: station score {station_score} < {self.min_station_score}"
                    )
                    continue
                else:
                    logger.info(
                        f"Accepted: station score {station_score} ≥ {self.min_station_score}"
                    )
                    final_catalog += cat
                    continue

            # Fallback: use minimum phase and P+S station criteria
            if o.quality.used_phase_count < self.nll_min_phase:
                logger.debug(
                    f"Rejected: insufficient phases ({o.quality.used_phase_count} < {self.nll_min_phase})"
                )
                continue

            ps_station_count = self.check_stations_with_P_and_S(
                e, o, self.min_station_with_P_and_S
            )
            if ps_station_count < self.min_station_with_P_and_S:
                logger.info(
                    f"Rejected: only {ps_station_count}/{self.min_station_with_P_and_S} stations with both P and S, "
                    f"{o.quality.used_phase_count} phases, {o.quality.used_station_count} stations "
                )
                continue

            logger.info(
                f"Accepted: {o.quality.used_phase_count} phases, "
                f"{o.quality.used_station_count} stations, "
                f"{ps_station_count} with both P and S (min: {self.min_station_with_P_and_S})"
            )
            final_catalog += cat

        # sort events by time
        final_catalog.events = sorted(
            final_catalog.events, key=lambda e: e.preferred_origin().time
        )
        logger.info(f"Total accepted events: {len(final_catalog)}")
        return final_catalog

    def get_localisations_from_nllobs_dir(
        self, OBS_PATH: str, picks: List[Pick] = None, append: bool = True
    ) -> Catalog:
        """nll localisation and export to quakeml

        warning : network and channel are lost since they are not used by nll
        use Phase() to get them back.

        Args:
            OBS_PATH (string): directory where are the nll obs files
            append (bool, optional): append new origin to self.catalog. Defaults to True.

        Returns:
            Catalog: returns a catalog of all computed origins
        """
        fallback_loc_method = "GAU_ANALYTIC"  # "EDT_OT_WT_ML"
        obs_files_pattern = os.path.join(OBS_PATH, "cluster-*.obs")
        logger.debug(f"Localization of {obs_files_pattern}")

        cat_results = []

        for i, nll_obs_file in enumerate(
            sorted(glob.glob(obs_files_pattern), key=sort_by_cluster_file)
        ):
            picks_set = picks[i] if picks else None

            try:
                cat = self.nll_localisation(
                    nll_obs_file, picks=picks_set, double_pass=self.double_pass
                )
            except FileNotFoundError as e:
                logger.error(f"{e}")
                sys.exit(1)
            except LocalizationError as e:
                logger.warning(
                    f"{e} - trying with {fallback_loc_method} for {nll_obs_file}"
                )
                try:
                    cat = self.nll_localisation(
                        nll_obs_file,
                        picks=picks_set,
                        double_pass=self.double_pass,
                        force_loc_method=fallback_loc_method,
                    )
                    logger.info(
                        f"Localization succeeded with {fallback_loc_method} for {nll_obs_file}"
                    )
                except Exception as e:
                    logger.error(
                        f"Localization failed even with {fallback_loc_method}: {e}"
                    )
                    cat = None
            except Exception as e:
                logger.exception(
                    f"Unexpected localization error for {nll_obs_file}: {e}"
                )
                cat = None

            if not cat:
                logger.debug(f"No loc obtained for {nll_obs_file} :/")
                continue
            cat_results.append(cat)

        mycatalog = self.get_catalog_from_results(cat_results)

        if append:
            self.catalog += mycatalog

        return mycatalog

    @staticmethod
    def _remap_arrivals_to_picks(
        origin: Origin,
        source_picks: List[Pick],
        target_event: Event,
        update_phase_hint: bool = False,
    ) -> None:
        """
        Remap arrivals in origin to reference picks in target_event, or add missing picks.

        For each arrival in origin:
        - If a matching pick exists in target_event (by station, time), remap to it
        - If no matching pick exists, add a clone of the source pick to target_event

        Args:
            origin: The origin whose arrivals need to be remapped
            source_picks: The list of picks that arrivals currently reference
            target_event: The event containing the picks that arrivals should reference
            update_phase_hint: If True, update target pick phase_hint when source has
                a more specific phase (e.g., Pg > P). Used for second pass relabeling.
        """
        # Build a lookup table from target picks: (station, time) -> target_pick
        # Note: We don't include phase in the key because relabeling may change it
        target_pick_map = {}
        for pick in target_event.picks:
            if pick.waveform_id is None:
                continue
            key = (
                pick.waveform_id.get_seed_string(),
                round(pick.time.timestamp, 6),
            )
            target_pick_map[key] = pick

        # Build a lookup table from source picks: pick_id -> (key, pick)
        source_pick_map = {}
        for pick in source_picks:
            if pick.waveform_id is None:
                continue
            key = (
                pick.waveform_id.get_seed_string(),
                round(pick.time.timestamp, 6),
            )
            source_pick_map[pick.resource_id.id] = (key, pick)

        # Remap each arrival
        for arrival in origin.arrivals:
            if arrival.pick_id is None:
                continue

            source_pick_id = arrival.pick_id.id
            if source_pick_id not in source_pick_map:
                logger.warning(
                    f"Arrival {arrival.resource_id} references unknown pick {source_pick_id}"
                )
                continue

            key, source_pick = source_pick_map[source_pick_id]
            if key in target_pick_map:
                target_pick = target_pick_map[key]
                arrival.pick_id = target_pick.resource_id
                # Update target pick phase_hint ONLY if requested and source has
                # a more specific phase (e.g., source has Pg/Sg from relabeling,
                # target has generic P/S)
                if (
                    update_phase_hint
                    and source_pick.phase_hint != target_pick.phase_hint
                    and len(source_pick.phase_hint) > len(target_pick.phase_hint)
                ):
                    target_pick.phase_hint = source_pick.phase_hint
            else:
                # No matching pick found: clone source pick with fresh id
                cloned_pick = copy.deepcopy(source_pick)
                base_id = (
                    source_pick.resource_id.id
                    if source_pick.resource_id
                    else f"smi:local/{uuid.uuid4()}"
                )
                cloned_pick.resource_id = ResourceIdentifier(
                    f"{base_id}-clone-{uuid.uuid4()}"
                )
                target_event.picks.append(cloned_pick)
                target_pick_map[key] = cloned_pick
                arrival.pick_id = cloned_pick.resource_id
                logger.debug(
                    "Cloned missing pick %s into target event for %s",
                    base_id,
                    key,
                )

    @staticmethod
    def _remap_arrivals_to_existing_picks(
        origin: Origin, source_event: Event, target_event: Event
    ) -> None:
        """
        Remap arrivals in origin to reference picks in target_event instead of source_event.

        Wrapper around _remap_arrivals_to_picks that extracts picks from source_event
        and enables phase_hint updates for second pass relabeling.

        Args:
            origin: The origin whose arrivals need to be remapped
            source_event: The event containing the picks that arrivals currently reference
            target_event: The event containing the picks that arrivals should reference
        """
        NllLoc._remap_arrivals_to_picks(
            origin, source_event.picks, target_event, update_phase_hint=True
        )

    def unset_arrival_gap_dist_km(self, event: Event, gap_dist_max_km) -> Event:
        """Unset arrival with gap in distance > dist_max

        Args:
            event (Event): event to work on

        Returns:
            Event: modified event with arrival time_weight set to 0 if gap_dist_max_km >= gap_dist_max_km
        """
        if gap_dist_max_km is None:
            logger.info("No gap_dist_max_km defined. Skip unset arrival.")
            return event

        arrivals_to_unset = get_arrival_with_distance_gap_greater_than(
            event, gap_dist_max_km
        )
        logger.info(
            f"Unset arrival time_weight due to gap_dist_max_km >= {gap_dist_max_km} km: {len(arrivals_to_unset)} arrivals."
        )

        for a in arrivals_to_unset:
            pick = get_pick_from_arrival(event, a)
            if pick is None:
                logger.error(f"Can't find pick for arrival {a.pick_id}")
                continue

            logger.debug(
                f"Unset arrival time_weight due to gap_dist_max_km >= ({gap_dist_max_km} km): "
                f"{pick.waveform_id.get_seed_string()} {a.phase} {pick.time}"
            )

            # find the corresponding arrival and set the weight to 0
            for arrival in event.preferred_origin().arrivals:
                if arrival.pick_id == pick.resource_id:
                    arrival.time_weight = 0
                    break

        return event

    def cleanup_pick_phase(self, event: Event) -> Event:
        """
        Remove picks/arrivals

        Remove picks/arrivals with:
            - time weight set to 0
            - bad residual
            - duplicated phases (remove the one with highest residual)
            - distance > dist_km_cutoff (if defined)

        Keep (forced):
            - pick with evaluation_mode set "manual" if keep_manual_picks is True
            - bypass relabel steps

        Update "used_station_count" and "used_phase_count" in origin quality.

        Args:
            event (Event): event to work on

        Returns:
            Event: modified event

        """
        orig = event.preferred_origin()
        pick_to_delete = []
        arrival_to_delete = []
        for arrival in orig.arrivals:
            pick = next(
                (p for p in event.picks if p.resource_id == arrival.pick_id), None
            )
            if pick is None:
                logger.error(f"Can't find pick for arrival {arrival.pick_id}")
                continue

            if self.keep_manual_picks and pick.evaluation_mode == "manual":
                continue

            if "P" in arrival.phase.upper():
                time_residual_threshold = self.P_time_residual_threshold
            elif "S" in arrival.phase.upper():
                time_residual_threshold = self.S_time_residual_threshold
            else:
                logger.warning(f"cleanup_pick_phase: unknown phase {arrival.phase}")
                time_residual_threshold = None

            bad_time_residual = (
                False
                if not time_residual_threshold
                else (fabs(arrival.time_residual) > time_residual_threshold)
            )

            if (
                isclose(arrival.time_weight, 0, abs_tol=time_weight_tolerance)
                or bad_time_residual
                or (
                    self.dist_km_cutoff is not None
                    and arrival.distance > self.dist_km_cutoff / 111.0
                )
            ):
                pick_to_delete.append(pick)
                arrival_to_delete.append(arrival)

        if arrival_to_delete:
            logger.info(
                f"cleanup_pick_phase: removed {len(arrival_to_delete)} arrivals "
                f"(time_weight=0, bad residual, or cutoff)"
            )
            for a in arrival_to_delete:
                p = next((pk for pk in pick_to_delete if pk.resource_id == a.pick_id), None)
                if p:
                    logger.debug(
                        f"  - {p.waveform_id.get_seed_string()} {a.phase} {p.time}"
                    )

        for a in arrival_to_delete:
            orig.arrivals.remove(a)
        for p in pick_to_delete:
            event.picks.remove(p)

        # check duplicated picks
        duplicates_removed = 0
        pick_to_delete = []
        arrival_to_delete = []
        comb = combinations(orig.arrivals, 2)
        for a1, a2 in comb:
            p1 = get_pick_from_arrival(event, a1)
            p2 = get_pick_from_arrival(event, a2)
            if p1 == p2:
                continue

            if (
                a1.phase == a2.phase
                and p1.waveform_id.network_code == p2.waveform_id.network_code
                and p1.waveform_id.station_code == p2.waveform_id.station_code
            ):
                if a1.time_residual < a2.time_residual:
                    # remove a2 and p2
                    p = p2
                    a = a2
                else:
                    # remove a1 and p1
                    p = p1
                    a = a1
                logger.debug(
                    f"Duplicated pick detected [{p.waveform_id.get_seed_string()}, {a.phase}, {p.time}]... "
                    f"removing the one with highest residual"
                )
                if p not in pick_to_delete:
                    pick_to_delete.append(p)
                    duplicates_removed += 1
                if a not in arrival_to_delete:
                    arrival_to_delete.append(a)

        if duplicates_removed > 0:
            logger.info(
                f"cleanup_pick_phase: removed {duplicates_removed} duplicated arrivals"
            )

        for a in arrival_to_delete:
            orig.arrivals.remove(a)
        for p in pick_to_delete:
            event.picks.remove(p)

        # update "stations used" with weight > 0
        orig = event.preferred_origin()
        orig.quality.used_station_count = NllLoc.get_used_station_count(event, orig)
        orig.quality.used_phase_count = NllLoc.get_used_phase_count(event, orig)
        return event

    def cleanup_picks_and_relabel_picks(
        self, event: Event, zone: Zone, eval_threshold: float = 0.10
    ) -> Tuple[Event, dict]:
        """
        Cleans up picks and relabels them based on the defined zone and evaluation threshold.

        Parameters:
        -----------
        event : Event
            The seismic event containing picks and arrivals.
        zone : Zone
            The zone containing polygon definitions and sigma value for evaluation.
        eval_threshold : float, optional
            The threshold for evaluation score to determine if a pick should be relabeled or removed (default is 0.10).

        Returns:
        --------
        Tuple[Event, dict]
            A tuple containing the updated event and a dictionary of relabeled picks with their comments.


        List of status:
            * 'relabel': relabel done (ie. no conflicts, score above threshold)
            * 'set by user': already set by user in accordance with the polygon found (do not relabel it)
            * 'score too low': score is too low, nothing done
            * 'ignored: already set': pick is manual, conflicts with an already existing pick, do nothing
            * "removed: already set": pick is automatic, conflicts with an already existing pick, remove it
            * 'removed': pick is not within a polygon
        """

        # Minimum distance to epicenter to consider an arrival to be relabeled
        min_distance_to_epicenter = 0.25  # degrees

        df_polygons = zone.picks_delimiter
        sigma = zone.sigma

        cleaned_by_polygon = 0
        cleaned_by_nll = 0
        cleaned_by_gap_dist = 0
        cleaned_by_cutoff = 0

        if df_polygons.empty:
            logger.warning("No polygon defined in zone. Can't cleanup picks.")
            # ic(zone)
        else:
            region_name = df_polygons["region"].unique()[0]

        orig = event.preferred_origin()

        # Deduplicate arrivals that point to the same station with the same phase
        # before relabeling. This prevents conflicts when two arrivals (e.g., from
        # different channels HHZ/BHZ) both want to be relabeled to the same phase.
        seen_arrival_keys = {}
        deduplicated_arrivals = []
        for a in orig.arrivals:
            p = get_pick_from_arrival(event, a)
            if p is None:
                continue
            # Key: (network, station, phase) - keep first arrival for each
            arrival_key = (
                p.waveform_id.network_code,
                p.waveform_id.station_code,
                str(a.phase),
            )
            if arrival_key in seen_arrival_keys:
                logger.debug(
                    f"Pre-relabel dedup: removing duplicate arrival for {arrival_key}"
                )
                continue
            seen_arrival_keys[arrival_key] = a
            deduplicated_arrivals.append(a)

        if len(deduplicated_arrivals) != len(orig.arrivals):
            logger.info(
                f"cleanup_picks_and_relabel_picks: deduplicated {len(orig.arrivals) - len(deduplicated_arrivals)} "
                f"arrivals with same station/phase before relabeling"
            )
            orig.arrivals = deduplicated_arrivals

        pick_to_delete = []
        arrival_to_delete = []
        relabel = {}

        # for arrival in orig.arrivals:
        for arrival in sorted(orig.arrivals, key=sort_by_phase):
            pick = get_pick_from_arrival(event, arrival)
            if pick is None:
                logger.error(f"Can't find pick for arrival {arrival.pick_id}")
                continue

            # remove pick with time_weight set to 0
            if isclose(arrival.time_weight, 0, abs_tol=time_weight_tolerance):
                logger.debug(
                    f"Remove pick {pick.waveform_id.get_seed_string()} {arrival.phase} {pick.time} "
                    f"with time_weight set to 0"
                )
                pick_to_delete.append(pick)
                arrival_to_delete.append(arrival)
                cleaned_by_nll += 1
                continue

            # remove pick with distance > dist_km_cutoff
            if (
                self.dist_km_cutoff is not None
                and arrival.distance > self.dist_km_cutoff / 111.0
            ):
                logger.info(
                    f"Remove pick {pick.waveform_id.get_seed_string()} {arrival.phase} {pick.time} "
                    f"with distance > {self.dist_km_cutoff} km"
                )
                pick_to_delete.append(pick)
                arrival_to_delete.append(arrival)
                cleaned_by_cutoff += 1
                continue

            if df_polygons.empty:
                continue

            # check if pick is within zone
            if arrival.phase in ["P", "S", "Pg", "Pn", "Sg", "Sn"]:
                key, score, polygons_score, evaluation_score = (
                    get_best_polygon_for_point(
                        Point(arrival.distance, pick.time - orig.time),
                        f"{pick.waveform_id.get_seed_string()} {arrival.phase}",
                        df_polygons,
                        sigma,
                        eval_threshold=eval_threshold,
                    )
                )

                # Check if pick is not within a polygon: remove it
                if len(polygons_score) == 0:
                    logger.debug(
                        f"Pick {pick.waveform_id.get_seed_string()} {arrival.phase} {pick.time}. "
                        f"has no polygon defined in {region_name}. Removing it."
                    )
                    # add comment to arrival, and keep track of it
                    relabel_key, comment = relabel_phase_and_comment_arrival(
                        arrival,
                        pick,
                        key,
                        evaluation_score,
                        polygons_score,
                        "removed",
                    )
                    relabel[relabel_key] = comment
                    pick_to_delete.append(pick)
                    arrival_to_delete.append(arrival)
                    cleaned_by_polygon += 1
                    # Fixme: keep arrival but set time_weight = 0 and propagate it to the next localization
                    # arrival.time_weight = 0
                    continue

                # User wants to filter out not well tagged phases but does not want to relabel them
                if not self.enable_relabel_pick_zone:
                    continue

                # Check if the station is too close to the epicenter
                # and if there is multiple phases.
                if (
                    min_distance_to_epicenter > 0
                    and arrival.distance < min_distance_to_epicenter
                    and len(polygons_score) > 1
                ):
                    original_phase = arrival.phase
                    logger.debug(
                        f"Pick {pick.waveform_id.get_seed_string()} {arrival.phase} {pick.time}. "
                        f"has a distance < {min_distance_to_epicenter} deg. Do nothing."
                    )
                    # add comment to arrival, and keep track of it
                    relabel_key, comment = relabel_phase_and_comment_arrival(
                        arrival,
                        pick,
                        original_phase,
                        evaluation_score,
                        polygons_score,
                        f"ignored: distance < {min_distance_to_epicenter} deg",
                    )
                    relabel[relabel_key] = comment
                    continue

                # Can't decide what to do
                if (key is None) and (score is None):
                    logger.debug(
                        f"Pick {pick.waveform_id.get_seed_string()} {arrival.phase} {pick.time}. "
                        f"Can't decide what to do (proba threshold is set to {eval_threshold}). "
                        f"Nothing to do."
                    )
                    # add comment to arrival, and keep track of it
                    relabel_key, comment = relabel_phase_and_comment_arrival(
                        arrival,
                        pick,
                        key,
                        evaluation_score,
                        polygons_score,
                        "score too low",
                    )
                    relabel[relabel_key] = comment
                    continue

                # Already set by user in accordance with the polygon found (do not relabel it)
                if key == arrival.phase:
                    logger.debug(
                        f"Pick {pick.waveform_id.get_seed_string()} {arrival.phase} {pick.time}. "
                        f"Selecting {key} zone (already set by user). Noting to do."
                    )
                    # add comment to arrival, and keep track of it
                    relabel_key, comment = relabel_phase_and_comment_arrival(
                        arrival,
                        pick,
                        key,
                        evaluation_score,
                        polygons_score,
                        "set by user",
                    )
                    relabel[relabel_key] = comment
                    continue

                # Check if the new label will not be
                # in conflict with an already existing one
                conflict = False
                for a in orig.arrivals:
                    if a == arrival:
                        continue
                    p = get_pick_from_arrival(event, a)
                    if (p.waveform_id.network_code, p.waveform_id.station_code) == (
                        pick.waveform_id.network_code,
                        pick.waveform_id.station_code,
                    ):
                        logger.debug(
                            f"Conflict check: current={pick.waveform_id.get_seed_string()} {arrival.phase} "
                            f"wants key={key}, other arrival has phase={a.phase}"
                        )
                        if key == a.phase:
                            logger.debug(
                                f"Pick {pick.waveform_id.get_seed_string()} {arrival.phase} {pick.time}. "
                                f"has a conflict with an arrival already set. Noting to do."
                            )
                            conflict = True
                            break

                if conflict:
                    original_phase = arrival.phase

                    # if the current arrival corresponds to an automatic pick, we remove it
                    # otherwise it is a manual so we just ignore it
                    if pick.evaluation_mode in [
                        "automatic",
                        None,
                    ] and arrival.phase in ["P", "S"]:
                        # do not remove picks and arrivals here
                        # but rather disable them later
                        # pick_to_delete.append(pick)
                        # arrival_to_delete.append(arrival)
                        action = "removed: already set"
                    else:
                        action = "ignored: already set"

                    # add comment to arrival, and keep track of it
                    relabel_key, comment = relabel_phase_and_comment_arrival(
                        arrival,
                        pick,
                        original_phase,
                        evaluation_score,
                        polygons_score,
                        force_status=action,
                    )
                    relabel[relabel_key] = comment
                    continue

                # Relabel pick
                logger.debug(
                    f"Pick {pick.waveform_id.get_seed_string()} {arrival.phase} {pick.time}. "
                    f"is within {key} zone. Relabeling it."
                )

                # add comment to arrival, and keep track of it
                relabel_key, comment = relabel_phase_and_comment_arrival(
                    arrival, pick, key, evaluation_score, polygons_score
                )
                relabel[relabel_key] = comment

        # remove picks and arrivals
        for a in arrival_to_delete:
            orig.arrivals.remove(a)
        for p in pick_to_delete:
            event.picks.remove(p)

        logger.info(
            f"Removed arrivals with time_weight set to 0 "
            f"(by nll ({cleaned_by_nll}), cutoff ({cleaned_by_cutoff}) or polygons ({cleaned_by_polygon})): "
            f"{len(arrival_to_delete)} arrivals"
        )

        # update "stations used" with weight > 0
        orig.quality.used_station_count = NllLoc.get_used_station_count(event, orig)
        orig.quality.used_phase_count = NllLoc.get_used_phase_count(event, orig)
        return event, relabel

    @staticmethod
    def get_used_station_count(event: Event, origin: Origin) -> int:
        """
        Calculates the number of unique stations used in the given origin.

        Parameters:
            event (Event): The event object.
            origin (Origin): The origin object.

        Returns:
            int: The number of unique stations used.
        """
        station_list = []
        for arrival in origin.arrivals:
            # if arrival.time_weight and arrival.time_residual:
            if hasattr(arrival, "time_weight") and not isclose(
                arrival.time_weight, 0, abs_tol=time_weight_tolerance
            ):
                pick = get_pick_from_arrival(event, arrival)
                if pick:
                    station_list.append(
                        f"{pick.waveform_id.network_code}.{pick.waveform_id.station_code}"
                    )
        return len(set(station_list))

    @staticmethod
    def get_used_phase_count(event: Event, origin: Origin) -> int:
        """
        Calculate the number of used phases for a given origin.

        Parameters:
            event (Event): The event object.
            origin (Origin): The origin object.

        Returns:
            int: The number of used phases.

        """
        nb_phase_used = 0
        for arrival in origin.arrivals:
            # if arrival.time_weight and arrival.time_residual:
            if hasattr(arrival, "time_weight") and not isclose(
                arrival.time_weight, 0, abs_tol=time_weight_tolerance
            ):
                pick = get_pick_from_arrival(event, arrival)
                if pick:
                    nb_phase_used += 1
        return nb_phase_used

    @staticmethod
    def replace(templatefile: str, outfilename: str, tags: dict) -> None:
        """
        Replace tags in a template file and write the result to an output file.

        Args:
            templatefile (str): The path to the template file.
            outfilename (str): The path to the output file.
            tags (dict): A dictionary containing the tags to be replaced in the template.

        Returns:
            None
        """
        try:
            with open(templatefile, encoding="utf-8") as file_:
                template = Template(file_.read())
        except FileNotFoundError:
            logger.error(f"Template file {templatefile} not found.")
            raise FileNotFoundError(
                f"Template file {templatefile} not found. Please check the path."
            )

        t = template.render(tags)
        with open(outfilename, "w") as out_fh:
            out_fh.write(t)
            logger.debug(f"Template {templatefile} rendered as {outfilename}")

    def show_localizations(self, output: str = "stdout", log_level: int = logging.INFO) -> None:
        """Show all localizations in the catalog.

        Args:
            output: Output destination, either "stdout" or "logger"
            log_level: Logging level to use when output="logger" (default: logging.INFO)
        """
        lines = []
        lines.append("%d events in catalog:" % len(self.catalog))
        lines.append("Text, T0, lat, lon, depth(m), RMS, sta_count, phase_count, gap1, gap2")
        for e in self.catalog.events:
            try:
                nll_obs = self.event_cluster_mapping[e.resource_id.id]
            except KeyError:
                nll_obs = ""
            lines.extend(format_event(e, nll_obs))

        # Output the buffer
        if output == "logger":
            for line in lines:
                logger.log(log_level, line)
        else:
            for line in lines:
                print(line)


def get_pick_from_arrival(event: Event, arrival: Arrival) -> Pick:
    """
    Retrieve a Pick object from given Arrival in a given Event.
    This function searches through the picks associated with the given event
    and returns the pick that matches the resource ID specified in the arrival.
    Args:
        event (Event): The event containing a list of picks.
        arrival (Arrival): The arrival containing the pick ID to search for.
    Returns:
        Pick: The pick that matches the arrival's pick ID, or None if no match is found.
    """
    pick = next((p for p in event.picks if p.resource_id == arrival.pick_id), None)
    return pick


def format_event(event: Event, txt: str = "", header: bool = False) -> List[str]:
    """Format event information as a list of strings.

    Args:
        event: The event to format
        txt: Text prefix for the event
        header: Whether to include a header line

    Returns:
        List of formatted strings
    """
    lines = []
    if header:
        lines.append(
            "Text, T0, lat, lon, depth, RMS, sta_count, phase_count, gap1, gap2, model, locator"
        )

    o_pref = event.preferred_origin()

    if hasattr(event, "event_type") and event.event_type == "not existing":
        lines.append(format_origin(o_pref, "FAKE"))
    else:
        lines.append(format_origin(o_pref, txt))

    for o in event.origins:
        if o == o_pref:
            continue
        lines.append(format_origin(o, " |__"))

    return lines


def show_event(event: Event, txt: str = "", header: bool = False):
    """Print event information to stdout."""
    for line in format_event(event, txt, header):
        print(line)


def format_origin(o: Origin, txt: str) -> str:
    """Format origin information as a string.

    Args:
        o: The origin to format
        txt: Text prefix for the origin

    Returns:
        Formatted string
    """
    azimuthal_gap = o.get("quality", {}).get("azimuthal_gap", None)
    if not azimuthal_gap:
        azimuths = [
            arrival.azimuth for arrival in o.arrivals if arrival.time_weight > 0
        ]
        azimuthal_gap = compute_azimuthal_gap(azimuths)
        if azimuthal_gap:
            azimuthal_gap = f"{azimuthal_gap:.1f}"
        else:
            azimuthal_gap = "-"
    else:
        azimuthal_gap = f"{azimuthal_gap:.1f}"

    secondary_azimuthal_gap = o.get("quality", {}).get("secondary_azimuthal_gap", None)
    if not secondary_azimuthal_gap:
        azimuths = [
            arrival.azimuth for arrival in o.arrivals if arrival.time_weight > 0
        ]
        secondary_azimuthal_gap = compute_secondary_azimuthal_gap(azimuths)
        if secondary_azimuthal_gap:
            secondary_azimuthal_gap = f"{secondary_azimuthal_gap:.1f}"
        else:
            secondary_azimuthal_gap = "-"
    else:
        secondary_azimuthal_gap = f"{secondary_azimuthal_gap:.1f}"

    return ", ".join(
        map(
            str,
            [
                txt,
                o.time,
                f"{o.latitude:.3f}",
                f"{o.longitude:.3f}",
                f"{o.depth:.1f}",
                (
                    f"{o.quality.standard_error:.3f}"
                    if o.quality.standard_error
                    else "-"
                ),
                o.quality.used_station_count,
                o.quality.used_phase_count,
                azimuthal_gap,
                secondary_azimuthal_gap,
                o.earth_model_id.id.split("/")[-1] if o.earth_model_id else "",
                o.method_id.id.split("/")[-1] if o.method_id else "",
            ],
        )
    )


def show_origin(o: Origin, txt: str) -> None:
    """Print origin information to stdout."""
    print(format_origin(o, txt))


def show_bulletin(
    event: Event, origin_id: ResourceIdentifier = None, zones: Zones = None, plot=False
) -> None:
    """
    Display a bulletin containing information about event origins and arrivals.

    Parameters:
        event (Event): The event object.
        origin_id (ResourceIdentifier, optional): The ID of the origin. Defaults to None.
        zones (Zones, optional): The zones object. Defaults to None.
        plot (bool, optional): Whether to plot the arrival time. Defaults to False.
    """

    if not origin_id:
        origin = event.preferred_origin()
    else:
        for o in event.origins:
            if o.resource_id == origin_id:
                origin = o
                break
        else:
            raise ValueError(f"Origin with id {origin_id} not found")

    # get the region name and polygon
    if zones:
        zone, _ = zones.find_zone(origin.latitude, origin.longitude)
        df_polygons = zone.picks_delimiter
    else:
        df_polygons = pd.DataFrame()

    table = PrettyTable()
    table.field_names = [
        "used",
        "station",
        "phase",
        "weight",
        "residual",
        "dist(deg)",
        "time",
        "uncertainty",
        "evaluation",
        "proba",
        "relabel",
    ]
    table.align["station"] = "l"
    table.align["phase"] = "l"
    table.align["relabel"] = "l"

    # print("station phase weight residual distance time evaluation")
    for arrival in origin.arrivals:
        pick = get_pick_from_arrival(event, arrival)
        if pick is None:
            logger.error(f"Can't find pick for arrival {arrival.pick_id}")
            logger.debug(f"arrival: {arrival}")
            continue

        if hasattr(arrival, "time_weight") and isclose(
            arrival.time_weight, 0, abs_tol=time_weight_tolerance
        ):
            used = False
        else:
            used = True

        wfid = pick.waveform_id
        station_name = f"{wfid.network_code}.{wfid.station_code}"
        phase_name = arrival.phase

        # Get from arrival comments:
        #  - relabel info
        # {
        #   "relabel": {
        #     "action": "set by user",
        #     "eval_score": "0.9970",
        #     "scores": {
        #       "Pn": "0.0003",
        #       "Sg": "0.9083",
        #       "Sn": "0.0028"
        #     }
        #   }
        # }

        # Get from pick comments:
        #  - pick probability
        # {"probability": {"name": "RENASS", "value": 0.92}}

        relabel = ""
        for c in arrival.comments:
            try:
                info = json.loads(c.text)
            except (json.JSONDecodeError, TypeError):
                continue
            relabel = ""
            if "relabel" in info.keys():
                phases_info = ""
                for k, v in info["relabel"]["scores"].items():
                    phases_info += f"{k}={v}, "
                relabel = (
                    f'action: {info["relabel"]["action"]} on {info["relabel"]["prev_phase"]},'
                    f'score: {info["relabel"]["eval_score"]}, {phases_info}'
                )

        probability = ""
        for c in pick.comments:
            try:
                info = json.loads(c.text)
            except (json.JSONDecodeError, TypeError):
                continue
            # decode : {'probability': {'name': 'RENASS', 'value': 0.68}}
            if "probability" in info.keys():
                probability = info["probability"]["value"]

        table.add_row(
            [
                used,
                station_name,
                phase_name,
                f"{arrival.time_weight:.2f}",
                f"{arrival.time_residual:.2f}",
                f"{arrival.distance:.3f}" if arrival.distance else "-",
                pick.time,
                pick.time_errors.uncertainty if hasattr(pick, "time_errors") else "-",
                pick.evaluation_mode,
                probability,
                relabel,
            ]
        )
        # print(f"{station_name} {phase_name} {arrival.time_weight} {arrival.time_residual} {arrival.distance} {pick.time} {pick.evaluation_mode}")

    # print(Event.__str__(event))
    
    print("\nQuality classification:")
    try:
        Q, QS, QD, classif_txt = classify_event(event, debug=True)
    except Exception as e:
        logger.error(f"\tError in classify_event: {e}")
        Q = 0
        QS = 0
        QD = 0
        classif_txt = "unknown"
    print(f"\thypo7 quality: {Q} ({classif_txt}), QS={QS}, QD={QD}")
    
    try:
        mlq = classify_event_michele_mod2(event)
    except Exception as e:
        logger.error(f"\tError in classify_event_michele_mod2: {e}")
        mlq = ("N/A", "N/A")
    print(f"\tMichele mod2 quality: Q={mlq[1]}, QF={mlq[0]:.2f}")
    
    print("\n")
    print(table)

    # plot with plotext library arrival time with respect to distance
    title = f"lat={origin.latitude:.3f}, lon={origin.longitude:.3f}, depth={origin.depth/1000.:.1f} km"
    if zones:
        title += f", {zone.velocity_profile} velocity model"

    if plot:
        plot_arrival_time(
            event=event, event_name=title, origin_id=origin_id, df_polygons=df_polygons
        )


def reloc_fdsn_event(
    locator: NllLoc,
    eventid: str = None,
    fdsnws: str = None,
    event: Event = None,
    zone_name: str = None,
) -> Catalog:
    """
    Retrieves earthquake event information from a FDSN web service or an event object
    and performs relocation using a locator object.

    Args:
        locator (Locator): The locator object used for event relocation.
        eventid (str): The ID of the earthquake event.
        fdsnws (str): The URL of the FDSN web service.
        event (Event): The earthquake event object.
        zone_name (str): The name of the zone to be used for relocation (forced).

    Returns:
        Catalog: A catalog object containing the relocated earthquake event.

    Raises:
        ValueError: If there is an error retrieving or reading the event information.
        ValueError: If the specified event ID does not exist.
        ValueError: If the specified model ID or template is not found.
    """

    if eventid is None and event is None:
        raise ValueError("No eventid or event provided.")

    if eventid:
        link = f"{fdsnws}/query?eventid={urllib.parse.quote(eventid, safe='')}&includearrivals=true"
        logger.debug(link)

        try:
            with urllib.request.urlopen(link, timeout=30) as f:
                cat = read_events(f.read())
        except Exception as e:
            raise ValueError(
                f"Error with {link}, can't get/read eventid {eventid} ({e})"
            ) from e

        if not cat:
            raise ValueError(f"[{eventid}] no such eventid !")

        event = cat[0]
    else:
        eventid = event.resource_id.id

    if locator.zones:
        # Set force_zone_name if provided
        if zone_name:
            logger.info(f"Forcing zone to {zone_name}.")
            locator.force_zone_name = zone_name

        origin = event.preferred_origin() or event.origins[0]
        zone, nll_template, model_id = locator._get_zone_and_template(
            origin.latitude, origin.longitude
        )

        if zone is None or len(zone) == 0:
            locator.zones.show_zones()
            raise ValueError(
                f"Zone not found for coordinates ({origin.latitude}, {origin.longitude})."
            )

        if locator.quakeml_settings and model_id:
            locator.quakeml_settings["model_id"] = model_id
        if nll_template:
            locator.nll_template = nll_template
        logger.info(f"Using zone '{zone['name']}' for event {eventid}.")
    else:
        default_model = (
            locator.quakeml_settings.get("model_id", "unknown")
            if locator.quakeml_settings
            else "unknown"
        )
        logger.warning(
            f"No zones defined, using default velocity model {default_model}."
        )

    try:
        cat = locator.reloc_event(event)
    except LocalizationError:
        raise
    except Exception as e:
        logger.exception(f"Unexpected localization error for event {eventid}: {e}")
        traceback.print_exc()
        raise

    return cat


def make_preloc_origin(
    o_parameters_file: str, picks_file: str, sta_file: str, quakeml_settings
) -> Tuple[Union[Origin, None], Union[Pick, None]]:
    """From PyOcto preliminary location build Origin/Arrivals/Picks

    Args:
        o_parameters_file (str): file with origin parameters
        picks_file (str): csv file with picks information
        sta_file (srt): csv file with station coordinates
        quakeml_settings (_type_): quakeml parameters to set up

    Returns:
        Tuple[Union[Origin, None], Union[Pick, None]]: Returns Origin, Picks objects
    """
    if not os.path.exists(o_parameters_file):
        logger.debug("Preloc: no preloc file")
        return None, None

    logger.debug(f"Preloc: reading {o_parameters_file} and {picks_file}")
    with open(o_parameters_file, encoding="utf-8") as vel:
        _ = vel.readline().strip()
        _ = vel.readline().strip()
        preloc_time = UTCDateTime(dateparser.parse(vel.readline().strip()))
        preloc_lat = float(vel.readline().strip())
        preloc_lon = float(vel.readline().strip())
        preloc_depth_m = float(vel.readline().strip())
        _ = float(vel.readline().strip())
        model_name_used = vel.readline().strip()
    logger.debug(
        f"Preloc: time={preloc_time}, lat={preloc_lat}, lon={preloc_lon}, "
        f"depth={preloc_depth_m}, model={model_name_used}"
    )

    preloc_origin = Origin()
    preloc_origin.evaluation_mode = "automatic"
    preloc_origin.evaluation_status = "preliminary"
    preloc_origin.method_id = ResourceIdentifier("PyOcto")
    preloc_origin.earth_model_id = ResourceIdentifier("haslach")
    if "agency_id" in quakeml_settings:
        preloc_origin.agency_id = quakeml_settings["agency_id"]
    else:
        preloc_origin.agency_id = "MyAgencyId"

    preloc_origin.time = preloc_time
    preloc_origin.latitude = preloc_lat
    preloc_origin.longitude = preloc_lon
    preloc_origin.depth = preloc_depth_m  # in meters
    preloc_origin.depth_type = "from location"
    preloc_origin.earth_model_id = model_name_used

    if "author" in quakeml_settings:
        author = quakeml_settings["author"]
    else:
        author = "DBClust"

    preloc_origin.creation_info = CreationInfo(
        creation_time=UTCDateTime.now(),
        agency_id=preloc_origin.agency_id,
        author=author,
        version="0",
    )

    # Read csv file and merge them on station column
    picks_df = pd.read_csv(picks_file)  # station, phase, time, residual
    coord_df = pd.read_csv(sta_file)  # id, latitude, longitude, elevation
    coord_df.drop_duplicates(inplace=True)
    coord_df.rename(columns={"id": "station"}, inplace=True)
    df = pd.merge(picks_df, coord_df, on="station", how="inner")

    preloc_origin.quality = OriginQuality()

    # preloc_origin.quality.used_phase_count = preloc_phase_count
    preloc_origin.quality.used_phase_count = len(df)
    preloc_origin.quality.associated_phase_count = (
        preloc_origin.quality.used_phase_count
    )
    preloc_origin.quality.used_station_count = (
        df["station"].apply(lambda x: ".".join(x.split(".")[:2])).nunique()
    )
    preloc_origin.quality.associated_station_count = (
        preloc_origin.quality.used_station_count
    )

    preloc_origin.quality.standard_error = np.round(
        np.sqrt((df["residual"] ** 2).mean()), 3
    )

    picks_list = []
    for r, row in df.iterrows():
        pick = Pick()
        pick.creation_info = CreationInfo(agency_id=preloc_origin.agency_id)
        # pick.evaluation_mode = "automatic"
        # pick.method_id = p.method
        parts = row["station"].split(".")
        net, sta = parts[:2] if len(parts) >= 2 else (parts[0] if parts else "", "")
        loc = parts[2] if len(parts) > 2 else ""
        chan = parts[3] if len(parts) > 3 else ""

        pick.waveform_id = WaveformStreamID(
            network_code=net,
            station_code=sta,
            location_code=loc,
            channel_code=chan,
        )
        pick.phase_hint = row["phase"]
        pick.time = row["time"]
        picks_list.append(pick)

        arrival = Arrival()
        arrival.phase = row["phase"]
        arrival.time_weight = 1
        arrival.time_residual = row["residual"]
        arrival.distance, arrival.azimuth, _ = gps2dist_azimuth(
            preloc_lat,
            preloc_lon,
            row["latitude"],
            row["longitude"],
        )
        # approx : convert to degres as distance is in meter
        arrival.distance = kilometer2degrees(arrival.distance / 1000.0)
        arrival.pick_id = pick.resource_id
        arrival.creation_info = CreationInfo(agencyID=preloc_origin.agency_id)
        preloc_origin.arrivals.append(arrival)

        # Fixme, add:
        # - pick manual|automatic

    distances = [a.distance for a in preloc_origin.arrivals]
    preloc_origin.minimum_distance = min(distances)
    preloc_origin.maximum_distance = max(distances)
    preloc_origin.median_distance = np.median(distances)
    azimuths = [a.azimuth for a in preloc_origin.arrivals]
    preloc_origin.quality.azimuthal_gap = compute_gap(azimuths)

    return preloc_origin, picks_list


if __name__ == "__main__":
    from config import DBClustConfig
    from dbclust import MyTemporaryDirectory

    logger.setLevel(logging.DEBUG)

    nlloc_bin = "NLLoc"
    scat2latlon_bin = "scat2latlon"
    nlloc_times_path = "/Users/marc/Dockers/routine/nll/data/times"
    nlloc_template = "../nll_template/nll_haslach-0.2_template.conf"
    tmpdir = "/tmp"

    conf = DBClustConfig(
        "/Users/marc/Data/DBClust/france.2016.01/dbclust-france.2016.01.yml"
    )

    zones = conf.zones
    enable_relabel_pick_zone = False
    enable_cleanup_pick_zone = True

    # eventid = "smi:local/437618f7-9cfe-4616-8e23-fdf32f7155db"
    # fdsnws = "http://localhost:10003/fdsnws/event/1"
    # nlloc_template = "../nll_template/nll_auvergne_template.conf"

    fdsnws = "https://api.franceseisme.fr/fdsnws/event/1"
    eventid = (
        "fr2023njqcnl"  # eost2023xcexglam : 6 relabels and a lot of automatic picks
    )
    eventid = "fr2023lznjuc"  # Lalaigne
    # eventid = "fr2023lojktv"
    # eventid = "fr2023mozdkg"  # 2 picks relabeled

    # fdsnws = "http://10.0.1.36:8080/fdsnws/event/1"
    # eventid = "eost2023dgdchbog"

    force_uncertainty = True
    P_uncertainty = 0.05
    S_uncertainty = 0.1

    # QuakeML settings: all must be defined
    quakeml_settings = {
        "agency_id": "RENASS",
        "author": "test@renass",
        "evaluation_mode": "automatic",
        "method_id": "NonLinLoc",
        "model_id": "haslach-0.2",
        # "model_id": "hybrid-pyrenees",
    }

    # cat = read_events("eost2023dgdchbog.qml")
    # e = cat[0]
    # for o in e.origins:
    #     if o.resource_id.id == "smi:org.gfz-potsdam.de/geofon/Origin/20230220085556.092564.256303":
    #         break
    # show_event(e, "****", header=True)
    # show_bulletin(e, zones)
    # sys.exit()

    with MyTemporaryDirectory(dir=tmpdir, delete=False) as tmp_path:
        locator = NllLoc(
            nlloc_bin,
            scat2latlon_bin,
            nlloc_times_path,
            nlloc_template,
            #
            # nll_obs_file=obs.name,
            tmpdir=tmp_path,
            #
            force_uncertainty=force_uncertainty,
            P_uncertainty=P_uncertainty,
            S_uncertainty=S_uncertainty,
            # dist_km_cutoff=None,  # KM
            # use_deactivated_arrivals=True,
            #
            double_pass=True,
            # P_time_residual_threshold=0.45,
            # S_time_residual_threshold=0.75,
            #
            quakeml_settings=quakeml_settings,
            #
            nll_verbose=False,
            #
            zones=zones,
            enable_relabel_pick_zone=enable_relabel_pick_zone,
            enable_cleanup_pick_zone=enable_cleanup_pick_zone,
            # polygon_proba_threshold=0.68,
            log_level=logging.INFO,
        )

        try:
            cat = reloc_fdsn_event(locator, eventid, fdsnws)
        except Exception as e:
            logger.error(f"Error with {eventid}: {e}")
            sys.exit()

        # check if there is only one event in the catalog
        if len(cat) != 1:
            logger.error("No event found or more than one event found !")
            sys.exit()

        event = cat[0]
        show_event(event, "****", header=True)
        show_bulletin(
            event=event,
            # origin_id="smi:org.gfz-potsdam.de/geofon/Origin/20230220085556.092564.256303",
            zones=zones,
        )

        cat.write(f"{urllib.parse.quote(eventid, safe='')}.qml", format="QUAKEML")
        cat.write(f"{urllib.parse.quote(eventid, safe='')}.sc3ml", format="SC3ML")
