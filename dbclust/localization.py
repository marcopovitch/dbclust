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
from pprint import pprint
from collections import defaultdict
from itertools import combinations
from math import fabs
from math import isclose
from math import isnan
from typing import List
from typing import Optional
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
from dbclust.gap import get_closest_station_dist_km
from dbclust.gap import get_station_count_before_distance_gap
from dbclust.gt5 import compute_gallacher_gt5_score_obspy
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
        closest_station_dist_km=None,
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
        min_ps_ratio: Optional[float] = None,  # Minimum S/P pick ratio (None = disabled)
        min_ps_ratio_wilson_z: Optional[float] = None,  # Wilson z for adaptive threshold (None = fixed)
        min_dist_relabel_deg: float = 0.0,  # minimum distance (degrees) to epicenter to allow relabeling
        min_time_weight: Optional[float] = None,  # remove all picks (incl. manual) with NLLoc time_weight below this threshold
        enable_residual_threshold_with_pick_zone: bool = False,  # apply P/S residual thresholds even when using pick zones
        pass2_degradation_factor: Optional[float] = None,  # warn if RMS_pass2 > RMS_pass1 * factor (None = disabled)
        pass2_fallback: bool = False,  # if True AND degradation detected, revert to pass 1 as preferred origin
        enable_time_weight_outlier_filter: bool = False,  # enable MAD-based outlier detection for abnormally low time_weight
        time_weight_outlier_mad_factor: float = 3.0,  # MAD multiplier for outlier threshold (higher = less aggressive)
        time_weight_outlier_min_picks: int = 5,  # minimum picks needed to compute MAD statistics
        time_weight_outlier_absolute_threshold: Optional[float] = None,  # absolute threshold for time_weight (regardless of MAD)
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
        self.closest_station_dist_km = closest_station_dist_km
        self.min_ps_ratio = min_ps_ratio
        self.min_ps_ratio_wilson_z = min_ps_ratio_wilson_z
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
        self.min_dist_relabel_deg = min_dist_relabel_deg
        self.min_time_weight = min_time_weight
        self.enable_residual_threshold_with_pick_zone = enable_residual_threshold_with_pick_zone
        self.pass2_degradation_factor = pass2_degradation_factor
        self.pass2_fallback = pass2_fallback
        self.enable_time_weight_outlier_filter = enable_time_weight_outlier_filter
        self.time_weight_outlier_mad_factor = time_weight_outlier_mad_factor
        self.time_weight_outlier_min_picks = time_weight_outlier_min_picks
        self.time_weight_outlier_absolute_threshold = time_weight_outlier_absolute_threshold

        logger.info(
            f"NllLoc initialized: MAD filter={'enabled' if enable_time_weight_outlier_filter else 'disabled'}, "
            f"factor={time_weight_outlier_mad_factor}, min_picks={time_weight_outlier_min_picks}, "
            f"absolute_threshold={time_weight_outlier_absolute_threshold}"
        )

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
        _pass_label: str = None,  # optional label for logging (e.g. "Pass 0 preloc")
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

                # Pass 0: fast NLL preloc with all raw picks to detect MAD outliers.
                # Only when double_pass is active — otherwise pass 1 is the only loc.
                if self.double_pass:
                    logger.info("Pass 0 (preloc NLL): GAU_ANALYTIC with all raw picks.")
                    # Use pass_count=1 to block double-pass recursion inside this call.
                    # Do NOT pass double_pass=False: that would mutate self.double_pass
                    # and break the pass 1 → pass 2 chain that follows.
                    cat_preloc = self.nll_localisation(
                        nll_obs_file,
                        picks=picks,
                        pass_count=1,  # blocks double-pass recursion without mutating self.double_pass
                        force_template=nll_template,
                        force_loc_method="GAU_ANALYTIC",
                        _pass_label="Pass 0 (preloc NLL/GAU_ANALYTIC)",
                    )
                    if not cat_preloc:
                        logger.warning("Pass 0 preloc NLL failed — aborting.")
                        return Catalog()

                    e_preloc = cat_preloc.events[0]
                    o_preloc = e_preloc.preferred_origin()
                    outlier_arrivals = self._detect_mad_outliers(e_preloc, o_preloc)

                    if outlier_arrivals:
                        if picks is None:
                            logger.warning(
                                "Pass 0 MAD filter: outliers detected but no picks list available "
                                "— cannot rewrite obs file, filter skipped."
                            )
                        else:
                            outlier_pick_ids = {a.pick_id for a in outlier_arrivals}
                            n_before = len(picks)
                            picks = [p for p in picks if p.resource_id not in outlier_pick_ids]
                            logger.info(
                                f"Pass 0 MAD filter: removed {n_before - len(picks)} outlier pick(s), "
                                f"{len(picks)} remaining."
                            )
                            # Rewrite the obs file from the original filtered picks list
                            # (not from e_preloc which may have NLL-merged co-located picks).
                            tmp_event = Event(picks=picks)
                            tmp_cat = Catalog(events=[tmp_event])
                            tmp_cat.write(nll_obs_file, format="NLLOC_OBS")

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
                raise LocalizationError(f"calc_maximum_likelihood_ot failed using {loc_method_used} method.")
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
        ps_with_both, ps_total, ps_ratio = self._compute_ps_ratio(e, o)
        o.quality.ps_station_count = ps_with_both

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
        event_ids_in_picks = sorted(set(
            p.creation_info.author.split("/")[-1]
            for p in e.picks
            if p.creation_info and p.creation_info.author
        ))
        pass_label = _pass_label if _pass_label else f"Pass {pass_count + 1}"
        logger.info(
            f"{pass_label} localization: "
            f"lat={o.latitude:.4f}, lon={o.longitude:.4f}, depth={o.depth/1000:.1f}km, "
            f"RMS={o.quality.standard_error:.3f}, phases={o.quality.used_phase_count}, "
            f"model={model_id}"
            + (f" [event_ids: {event_ids_in_picks}]" if event_ids_in_picks else "")
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
            # Update template/model_id for second pass based on NLL first-pass position.
            # The preloc (.vel) guided the first pass, but the NLL result may fall in a different zone.
            # We always trust the NLL-derived zone for the second pass (unless zone or template is forced).
            zone_is_valid = zone is not None and len(zone) > 0
            if (
                zone_is_valid
                and not self.force_zone_name
                and not force_template
                and zone_template
            ):
                # Use detected zone's template for the second pass (even if preloc existed)
                nll_template = zone_template
                model_id = zone_model_id
                logger.info(
                    f"Second pass: using zone '{zone['name']}' template: {nll_template}, model: {model_id}"
                    + (" [overrides preloc zone]" if os.path.exists(vel_file) else "")
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

            # Sync event2 back into cat2 (cleanup may have replaced the object)
            cat2.events[0] = event2

            if len(event2.picks):
                new_nll_obs_file = nll_obs_file + ".2nd_pass"
                n_arrivals = len(event2.preferred_origin().arrivals)
                n_picks = len(event2.picks)
                logger.info(
                    f"Writing {n_picks} picks ({n_arrivals} arrivals) to NLLOC_OBS file for second pass"
                )
                if n_picks != n_arrivals:
                    orphan_picks = [
                        p for p in event2.picks
                        if not any(a.pick_id == p.resource_id for a in event2.preferred_origin().arrivals)
                    ]
                    logger.info(
                        f"Second pass: {n_picks - n_arrivals} orphan pick(s) without arrival "
                        f"(will be ignored by NLL): "
                        + ", ".join(f"{p.waveform_id.get_seed_string()} {p.phase_hint}" for p in orphan_picks)
                    )
                cat2.write(new_nll_obs_file, format="NLLOC_OBS")
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

                # Re-apply gap_dist_max_km check on pass 2 result.
                # Pass 1 location may be too noisy to reveal the real station gap;
                # pass 2 with the correct velocity model gives a reliable geometry.
                # Only evaluated for fully automatic events (any manual pick bypasses this).
                # Returns None if no gap or if manual picks are present.
                station_count_before_gap = get_station_count_before_distance_gap(
                    event2, self.gap_dist_max_km
                )
                if station_count_before_gap is not None:
                    arrivals_with_gap = get_arrival_with_distance_gap_greater_than(
                        event2, self.gap_dist_max_km
                    )
                    gap_stations = []
                    for a in arrivals_with_gap:
                        p = next((pk for pk in event2.picks if pk.resource_id == a.pick_id), None)
                        if p:
                            gap_stations.append(f"{p.waveform_id.get_seed_string()} {a.phase} dist={a.distance*111.1:.0f}km")
                    # FIXME: hardcoded threshold, should be passed as a NllLoc parameter
                    if station_count_before_gap < 5:
                        # Too few local stations support the solution: likely a fake event
                        # driven by distant stations with no nearby corroboration.
                        logger.warning(
                            f"Rejected (automatic picks only): only {station_count_before_gap} station(s) "
                            f"before gap_dist_max_km={self.gap_dist_max_km}km "
                            f"< 5 (hardcoded). Likely fake event. "
                            f"Arrivals beyond gap: {', '.join(gap_stations)}"
                        )
                        e.event_type = "not existing"
                        if not self.keep_not_existing_event:
                            return Catalog()
                    else:
                        logger.warning(
                            f"Pass 2 (automatic picks only): {len(arrivals_with_gap)} arrival(s) beyond "
                            f"gap_dist_max_km={self.gap_dist_max_km}km "
                            f"({station_count_before_gap} station(s) before gap): {', '.join(gap_stations)}"
                        )

                # add this new origin to catalog and set it as preferred
                # Remap arrivals in orig2 to reference picks in e.picks (not event2.picks)
                # because event2 has new pick IDs generated by obspy
                self._remap_arrivals_to_existing_picks(orig2, event2, e)
                e.origins.append(orig2)
                e.preferred_origin_id = orig2.resource_id

                # Detect pass 2 degradation: warn always, optionally revert to pass 1
                if self.pass2_degradation_factor is not None:
                    rms1 = o.quality.standard_error if (o and o.quality) else None
                    rms2 = orig2.quality.standard_error if (orig2 and orig2.quality) else None
                    if rms1 is not None and rms2 is not None and rms1 > 0:
                        if rms2 > rms1 * self.pass2_degradation_factor:
                            if self.pass2_fallback:
                                logger.warning(
                                    f"Pass 2 degradation detected: RMS pass1={rms1:.3f}s, "
                                    f"RMS pass2={rms2:.3f}s (factor={rms2/rms1:.1f} > "
                                    f"{self.pass2_degradation_factor}). "
                                    f"Reverting to pass 1 as preferred origin and removing pass 2 origin."
                                )
                                e.preferred_origin_id = o.resource_id
                                e.origins.remove(orig2)
                            else:
                                logger.warning(
                                    f"Pass 2 degradation detected: RMS pass1={rms1:.3f}s, "
                                    f"RMS pass2={rms2:.3f}s (factor={rms2/rms1:.1f} > "
                                    f"{self.pass2_degradation_factor}). "
                                    f"Keeping pass 2 as preferred origin (pass2_fallback=False)."
                                )
                        else:
                            logger.debug(
                                f"Pass 2 OK: RMS pass1={rms1:.3f}s, RMS pass2={rms2:.3f}s "
                                f"(factor={rms2/rms1:.1f})"
                            )
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

    def _compute_ps_ratio(self, event, origin) -> tuple:
        """Compute PS ratio (stations with both P and S / total stations).
        Returns (stations_with_both, total_stations, ps_ratio)."""
        station_phases = defaultdict(set)
        for arrival in origin.arrivals:
            if arrival.time_weight is None or arrival.time_weight == 0:
                continue
            pick = next(
                (p for p in event.picks if p.resource_id == arrival.pick_id), None
            )
            if pick is None or pick.waveform_id is None:
                continue
            station_code = (
                f"{pick.waveform_id.network_code}.{pick.waveform_id.station_code}"
            )
            phase = arrival.phase.lower() if arrival.phase else ""
            if phase.startswith("p"):
                station_phases[station_code].add("P")
            elif phase.startswith("s"):
                station_phases[station_code].add("S")

        total_stations = len(station_phases)
        stations_with_both = sum(
            1 for phases in station_phases.values() if "P" in phases and "S" in phases
        )
        ps_ratio = stations_with_both / total_stations if total_stations > 0 else 0.0
        return stations_with_both, total_stations, ps_ratio

    @staticmethod
    def _wilson_high(n_ps: int, n: int, z: float) -> float:
        """Wilson score interval upper bound.
        Used to test if observed ps_ratio is statistically below threshold.
        """
        if n <= 0:
            return 1.0
        p = min(n_ps / n, 1.0)
        denom = 1 + z ** 2 / n
        center = p + z ** 2 / (2 * n)
        margin = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
        return (center + margin) / denom

    def _check_ps_ratio(self, event, origin) -> bool:
        """Check PS ratio (stations with both P and S / total stations).
        If min_ps_ratio_wilson_z is set, reject only when the Wilson upper bound
        of the observed ratio is below min_ps_ratio (adaptive, sample-size aware).
        Otherwise fall back to a fixed threshold comparison.
        Returns False (reject) if below threshold.
        """
        if self.min_ps_ratio is None:
            return True
        n_ps, n_total, ps_ratio = self._compute_ps_ratio(event, origin)
        if self.min_ps_ratio_wilson_z is not None:
            return self._wilson_high(n_ps, n_total, self.min_ps_ratio_wilson_z) >= self.min_ps_ratio
        return ps_ratio >= self.min_ps_ratio

    def get_catalog_from_results(self, cat_results: List[Catalog]) -> Catalog:
        """Compute attributes and filter events from catalogs"""
        final_catalog = Catalog()
        all_event_ids_seen: set = set()
        accepted_event_ids: set = set()
        for cat in cat_results:
            if not cat or not cat.events:
                logger.debug("Empty catalog or missing events, skipping.")
                continue

            # Each catalog is expected to have exactly one event
            e = cat.events[0]
            o = e.preferred_origin()

            # Extract event_ids from picks (stored in creation_info.author)
            event_ids_in_picks = sorted(set(
                p.creation_info.author.split("/")[-1]
                for p in e.picks
                if p.creation_info and p.creation_info.author
            ))
            all_event_ids_seen.update(event_ids_in_picks)
            event_ids_str = f" [{', '.join(event_ids_in_picks)}]" if event_ids_in_picks else ""

            # Compute quality attributes
            o.quality.used_station_count = self.get_used_station_count(e, o)
            o.quality.used_phase_count = self.get_used_phase_count(e, o)
            ps_with_both, ps_total, ps_ratio = self._compute_ps_ratio(e, o)
            o.quality.ps_station_count = ps_with_both

            # Gather all criteria values upfront for consolidated logging
            closest_km = get_closest_station_dist_km(e) if self.closest_station_dist_km is not None else None
            station_score = self.get_origin_station_score(e, o)
            ps_str = f"ps={ps_with_both}/{ps_total}({ps_ratio:.2f})"
            closest_str = f"closest={closest_km:.1f}km" if closest_km is not None else "closest=N/A"
            summary = (
                f"score={station_score}/{self.min_station_score} | "
                f"phases={o.quality.used_phase_count} | "
                f"stations={o.quality.used_station_count} | "
                f"{ps_str} | "
                f"{closest_str}"
                f"{event_ids_str}"
            )

            # reject event if closest station after final relocation is too far
            if self.closest_station_dist_km is not None and closest_km is not None and closest_km > self.closest_station_dist_km:
                log_fn = logger.warning if event_ids_in_picks else logger.info
                log_fn(f"Rejected | {summary} | reason: closest={closest_km:.1f}km > {self.closest_station_dist_km}km")
                continue

            if self.min_station_score is not None:
                if station_score < self.min_station_score:
                    log_fn = logger.warning if event_ids_in_picks else logger.info
                    log_fn(f"Rejected | {summary} | reason: score < {self.min_station_score}")
                    continue
                if self.min_ps_ratio is not None:
                    ps_rejected = (
                        self._wilson_high(ps_with_both, ps_total, self.min_ps_ratio_wilson_z) < self.min_ps_ratio
                        if self.min_ps_ratio_wilson_z is not None
                        else ps_ratio < self.min_ps_ratio
                    )
                    if ps_rejected:
                        if event_ids_in_picks:
                            logger.warning(
                                f"Accepted despite low ps_ratio | {summary} | "
                                f"reason: known event_id support"
                            )
                        else:
                            log_fn = logger.warning if event_ids_in_picks else logger.info
                            log_fn(
                                f"Rejected | {summary} | reason: ps_ratio={ps_ratio:.2f} < {self.min_ps_ratio}"
                            )
                            continue
                logger.info(f"Accepted | {summary}")
                accepted_event_ids.update(event_ids_in_picks)
                final_catalog += cat
                continue

            # Fallback: use minimum phase and P+S station criteria
            if o.quality.used_phase_count < self.nll_min_phase:
                log_fn = logger.warning if event_ids_in_picks else logger.debug
                log_fn(f"Rejected | {summary} | reason: phases={o.quality.used_phase_count} < {self.nll_min_phase}")
                continue

            ps_station_count = self.check_stations_with_P_and_S(e, o, self.min_station_with_P_and_S)
            if ps_station_count < self.min_station_with_P_and_S:
                log_fn = logger.warning if event_ids_in_picks else logger.info
                log_fn(f"Rejected | {summary} | reason: P+S stations={ps_station_count} < {self.min_station_with_P_and_S}")
                continue

            if self.min_ps_ratio is not None:
                ps_rejected = (
                    self._wilson_high(ps_with_both, ps_total, self.min_ps_ratio_wilson_z) < self.min_ps_ratio
                    if self.min_ps_ratio_wilson_z is not None
                    else ps_ratio < self.min_ps_ratio
                )
                if ps_rejected:
                    if event_ids_in_picks:
                        logger.warning(
                            f"Accepted despite low ps_ratio | {summary} | "
                            f"reason: known event_id support"
                        )
                    else:
                        log_fn = logger.warning if event_ids_in_picks else logger.info
                        log_fn(
                            f"Rejected | {summary} | reason: ps_ratio={ps_ratio:.2f} < {self.min_ps_ratio}"
                        )
                        continue

            logger.info(f"Accepted | {summary}")
            accepted_event_ids.update(event_ids_in_picks)
            final_catalog += cat

        # sort events by time
        final_catalog.events = sorted(
            final_catalog.events, key=lambda e: e.preferred_origin().time
        )
        logger.info(f"Total accepted events: {len(final_catalog)}")

        # Report known event_ids that did not make it into the final catalog
        lost_event_ids = all_event_ids_seen - accepted_event_ids
        if lost_event_ids:
            logger.warning(f"Known event_id(s) not found in accepted events: {sorted(lost_event_ids)}")

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

        nll_obs_files = sorted(glob.glob(obs_files_pattern), key=sort_by_cluster_file)
        n_total = len(nll_obs_files)
        for i, nll_obs_file in enumerate(nll_obs_files):
            picks_set = picks[i] if picks else None
            logger.info(f"--- Event #{i + 1}/{n_total} [{os.path.basename(nll_obs_file)}] ---")

            try:
                cat = self.nll_localisation(
                    nll_obs_file, picks=picks_set, double_pass=self.double_pass
                )
            except FileNotFoundError as e:
                logger.error(f"{e} - skipping {nll_obs_file}")
                cat = None
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
            - abnormally low time_weight (MAD-based outlier detection, if enabled)
            - bad residual
            - duplicated phases (remove the one with highest residual)
            - distance > dist_km_cutoff (if defined)

        Keep (forced):
            - pick with evaluation_mode set "manual" if keep_manual_picks is True
            - bypass relabel steps

        MAD outlier detection (if enable_time_weight_outlier_filter=True):
            - Computes median and MAD of time_weight values
            - Removes picks with time_weight < median - k*MAD
            - Only applied if number of picks >= time_weight_outlier_min_picks

        Update "used_station_count" and "used_phase_count" in origin quality.

        Args:
            event (Event): event to work on

        Returns:
            Event: modified event

        """
        orig = event.preferred_origin()
        pick_to_delete = []
        arrival_to_delete = []
        
        # Detect time_weight outliers using MAD (Median Absolute Deviation)
        mad_outliers = self._detect_mad_outliers(event, orig)
        arrival_to_delete.extend(mad_outliers)
        for _a in mad_outliers:
            _p = get_pick_from_arrival(event, _a)
            if _p is not None and _p not in pick_to_delete:
                pick_to_delete.append(_p)
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
                if pick not in pick_to_delete:
                    pick_to_delete.append(pick)
                if arrival not in arrival_to_delete:
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
            if a in orig.arrivals:
                orig.arrivals.remove(a)
        for p in pick_to_delete:
            if p in event.picks:
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
        min_distance_to_epicenter = self.min_dist_relabel_deg

        df_polygons = zone.picks_delimiter
        sigma = zone.sigma

        cleaned_by_polygon = 0
        cleaned_by_nll = 0
        cleaned_by_residual = 0
        cleaned_by_gap_dist = 0
        cleaned_by_cutoff = 0

        arrival_to_delete = []
        pick_to_delete = []

        if df_polygons.empty:
            logger.warning("No polygon defined in zone. Can't cleanup picks.")
            # ic(zone)
        else:
            region_name = df_polygons["region"].unique()[0]

        orig = event.preferred_origin()

        # Detect time_weight outliers using MAD (Median Absolute Deviation)
        # Do this BEFORE other filters to ensure outliers are counted correctly
        mad_outliers = self._detect_mad_outliers(event, orig)
        cleaned_by_mad = len(mad_outliers)
        logger.info(f"MAD outliers detected: {cleaned_by_mad} arrivals")
        arrival_to_delete.extend(mad_outliers)
        for _a in mad_outliers:
            _p = get_pick_from_arrival(event, _a)
            if _p is not None:
                pick_to_delete.append(_p)

        # Deduplicate arrivals that point to the same station with the same phase
        # before relabeling. This prevents conflicts when two arrivals (e.g., from
        # different channels HHZ/BHZ) both want to be relabeled to the same phase.
        # Sort by time_weight descending so the highest-weight arrival is kept.
        seen_arrival_keys = {}
        deduplicated_arrivals = []
        picks_to_remove_from_dedup = []
        for a in sorted(orig.arrivals, key=lambda x: x.time_weight or 0, reverse=True):
            p = get_pick_from_arrival(event, a)
            if p is None:
                continue
            # Key: (network, station, phase) - keep highest-weight arrival for each
            arrival_key = (
                p.waveform_id.network_code,
                p.waveform_id.station_code,
                str(a.phase),
            )
            if arrival_key in seen_arrival_keys:
                logger.debug(
                    f"Pre-relabel dedup: removing duplicate arrival for {arrival_key}"
                )
                picks_to_remove_from_dedup.append(p)
                continue
            seen_arrival_keys[arrival_key] = a
            deduplicated_arrivals.append(a)

        if len(deduplicated_arrivals) != len(orig.arrivals):
            logger.info(
                f"cleanup_picks_and_relabel_picks: deduplicated {len(orig.arrivals) - len(deduplicated_arrivals)} "
                f"arrivals with same station/phase before relabeling"
            )
            orig.arrivals = deduplicated_arrivals
            for p in picks_to_remove_from_dedup:
                if p in event.picks:
                    event.picks.remove(p)

        # Remove generic phase (S or P) when more specific phases of the same family
        # are present on the same station.
        #
        # Physics: S = first S arrival = min(t(Sg), t(Sn))
        #          P = first P arrival = min(t(Pg), t(Pn), t(Pb))
        #
        # Rules:
        #   - If S + Sg (no Sn): S is duplicate of Sg → keep best weight, remove other
        #   - If S + Sn (no Sg): S is duplicate of Sn → keep best weight, remove other
        #   - If S + Sg + Sn: S is duplicate of earliest specific phase (Sg or Sn)
        #     → compare weight(S) vs weight(earliest), keep best, remove other
        #   Same logic applies to P / Pg / Pn / Pb family.
        #
        # Note: if S (or P) has no specific counterpart, leave it alone — the zone
        # relabeling will rename it to Sg/Sn (or Pg/Pn) as appropriate.

        PHASE_FAMILIES = {
            "S": {"generic": "S", "specific": ["Sg", "Sn"]},
            "P": {"generic": "P", "specific": ["Pg", "Pn", "Pb"]},
        }

        # Build lookup: (net, sta, phase) -> (arrival, pick)
        arrival_pick_by_phase = {}
        for a in orig.arrivals:
            p = get_pick_from_arrival(event, a)
            if p is None:
                continue
            key = (
                p.waveform_id.network_code,
                p.waveform_id.station_code,
                str(a.phase),
            )
            arrival_pick_by_phase[key] = (a, p)

        generic_arrivals_to_remove = []
        specific_arrivals_to_remove = []

        for family in PHASE_FAMILIES.values():
            generic_phase = family["generic"]
            specific_phases = family["specific"]

            # Collect all stations that have the generic phase
            generic_keys = [
                k for k in arrival_pick_by_phase if k[2] == generic_phase
            ]

            for gkey in generic_keys:
                net, sta, _ = gkey
                generic_arrival, generic_pick = arrival_pick_by_phase[gkey]

                # Find specific phases present on this station
                specific_present = [
                    (arrival_pick_by_phase[(net, sta, sp)], sp)
                    for sp in specific_phases
                    if (net, sta, sp) in arrival_pick_by_phase
                ]

                if not specific_present:
                    # No specific phase → leave generic alone for zone relabeling
                    continue

                # Identify the earliest specific phase (S = min(t(Sg), t(Sn)))
                earliest_pair, earliest_phase = min(
                    specific_present,
                    key=lambda x: x[0][1].time,  # x[0] = (arrival, pick), [1] = pick
                )
                earliest_arrival, earliest_pick = earliest_pair

                gw = generic_arrival.time_weight or 0
                ew = earliest_arrival.time_weight or 0

                if gw > ew:
                    # Generic has better weight → remove the earliest specific phase
                    # (generic will be relabeled to that specific phase by zone relabeling)
                    logger.info(
                        f"Pre-relabel phase-family dedup: {net}.{sta} has {generic_phase} "
                        f"(w={gw}) + {earliest_phase} (w={ew}): removing {earliest_phase} "
                        f"(generic has better weight, will be relabeled)"
                    )
                    specific_arrivals_to_remove.append((earliest_arrival, earliest_pick))
                else:
                    # Specific has better or equal weight → remove generic
                    logger.info(
                        f"Pre-relabel phase-family dedup: {net}.{sta} has {generic_phase} "
                        f"(w={gw}) + {earliest_phase} (w={ew}): removing {generic_phase} "
                        f"(specific phase has better or equal weight)"
                    )
                    generic_arrivals_to_remove.append((generic_arrival, generic_pick))

        arrivals_to_remove_from_family_dedup = (
            generic_arrivals_to_remove + specific_arrivals_to_remove
        )
        if arrivals_to_remove_from_family_dedup:
            logger.info(
                f"Pre-relabel phase-family dedup: removing {len(arrivals_to_remove_from_family_dedup)} "
                f"redundant phase(s) from phase families"
            )
            for a, p in arrivals_to_remove_from_family_dedup:
                if a in orig.arrivals:
                    orig.arrivals.remove(a)
                if p in event.picks:
                    event.picks.remove(p)
        # Phase-family deduplication done            
        
        # After deduplication, relabel picks based on zone polygons
        # and remove picks/arrivals with:
        #   - time_weight set to 0
        #   - bad residual
        #   - duplicated phases (remove the one with highest residual)
        #   - distance > dist_km_cutoff (if defined)
        pick_to_delete = []
        # arrival_to_delete already initialized above with MAD outliers — do not reset
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

            # remove pick (including manual) with NLLoc time_weight below min_time_weight
            # catches clock-drift issues: NLLoc down-weights all phases from the affected station
            if (
                self.min_time_weight is not None
                and arrival.time_weight < self.min_time_weight
            ):
                logger.info(
                    f"Remove pick {pick.waveform_id.get_seed_string()} {arrival.phase} {pick.time} "
                    f"with time_weight={arrival.time_weight:.3f} < min_time_weight={self.min_time_weight} "
                    f"(possible clock issue)"
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

            # Optionally filter by time residual even when using pick zones
            if self.enable_residual_threshold_with_pick_zone:
                if "P" in arrival.phase.upper():
                    time_residual_threshold = self.P_time_residual_threshold
                elif "S" in arrival.phase.upper():
                    time_residual_threshold = self.S_time_residual_threshold
                else:
                    time_residual_threshold = None
                if time_residual_threshold and fabs(arrival.time_residual) > time_residual_threshold:
                    logger.info(
                        f"Remove pick {pick.waveform_id.get_seed_string()} {arrival.phase} "
                        f"time_residual={arrival.time_residual:.2f}s > threshold={time_residual_threshold}s"
                    )
                    pick_to_delete.append(pick)
                    arrival_to_delete.append(arrival)
                    cleaned_by_residual += 1
                    continue

            if df_polygons.empty:
                # No polygon defined for this zone: fallback to residual filtering
                if "P" in arrival.phase.upper():
                    time_residual_threshold = self.P_time_residual_threshold
                elif "S" in arrival.phase.upper():
                    time_residual_threshold = self.S_time_residual_threshold
                else:
                    time_residual_threshold = None
                if time_residual_threshold and fabs(arrival.time_residual) > time_residual_threshold:
                    logger.info(
                        f"Remove pick {pick.waveform_id.get_seed_string()} {arrival.phase} "
                        f"time_residual={arrival.time_residual:.2f}s > threshold={time_residual_threshold}s "
                        f"(no polygon defined, fallback to residual filter)"
                    )
                    pick_to_delete.append(pick)
                    arrival_to_delete.append(arrival)
                    cleaned_by_residual += 1
                continue

            # check if pick is within zone
            if arrival.phase in ["P", "S", "Pg", "Pn", "Sg", "Sn"]:
                try:
                    _dist_check = float(arrival.distance)
                    _dist_invalid = arrival.distance is None or isnan(_dist_check)
                except (TypeError, ValueError):
                    _dist_invalid = True
                if _dist_invalid:
                    logger.debug(
                        f"Pick {pick.waveform_id.get_seed_string()} {arrival.phase}: "
                        f"skipping polygon check (distance is None/NaN)"
                    )
                    continue
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
            if a in orig.arrivals:
                orig.arrivals.remove(a)
        for p in pick_to_delete:
            if p in event.picks:
                event.picks.remove(p)

        logger.info(
            f"Removed arrivals: nll/weight ({cleaned_by_nll}), residual ({cleaned_by_residual}), "
            f"cutoff ({cleaned_by_cutoff}), polygons ({cleaned_by_polygon}), MAD ({cleaned_by_mad}): "
            f"{len(arrival_to_delete)} total"
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

    def _detect_mad_outliers(self, event: Event, origin: Origin) -> list:
        """
        Detect time_weight outliers using Median Absolute Deviation (MAD).

        Returns list of arrivals whose time_weight is anomalously low.
        Only active when enable_time_weight_outlier_filter=True and there
        are at least time_weight_outlier_min_picks arrivals.
        Manual picks are excluded from the removal candidates when
        keep_manual_picks=True, but still included in the statistics.
        """
        outlier_arrivals = []

        if not self.enable_time_weight_outlier_filter:
            return outlier_arrivals

        logger.info(f"MAD filter enabled: checking {len(origin.arrivals)} arrivals")

        time_weights = []
        arrival_list = []
        for arr in origin.arrivals:
            pk = next((p for p in event.picks if p.resource_id == arr.pick_id), None)
            if pk is None:
                continue
            time_weights.append(arr.time_weight)
            if not (self.keep_manual_picks and pk.evaluation_mode == "manual"):
                arrival_list.append(arr)

        if len(time_weights) < self.time_weight_outlier_min_picks:
            return outlier_arrivals

        time_weights_array = np.array(time_weights)
        median_tw = np.median(time_weights_array)
        mad = np.median(np.abs(time_weights_array - median_tw))
        logger.info(
            f"MAD stats: n={len(time_weights)}, median={median_tw:.3f}, MAD={mad:.3f}, "
            f"weights={sorted(time_weights)[:5]}...{sorted(time_weights)[-3:]}"
        )

        if mad <= 0:
            return outlier_arrivals

        mad_threshold = median_tw - self.time_weight_outlier_mad_factor * mad
        if self.time_weight_outlier_absolute_threshold is not None:
            threshold = max(mad_threshold, self.time_weight_outlier_absolute_threshold)
            logger.info(
                f"MAD outlier detection: median={median_tw:.3f}, MAD={mad:.3f}, "
                f"mad_threshold={mad_threshold:.3f}, absolute_threshold={self.time_weight_outlier_absolute_threshold:.3f}, "
                f"final_threshold={threshold:.3f}"
            )
        else:
            threshold = mad_threshold
            logger.info(
                f"MAD outlier detection: median={median_tw:.3f}, MAD={mad:.3f}, "
                f"threshold={threshold:.3f}"
            )

        for arr in arrival_list:
            if arr.time_weight < threshold:
                outlier_arrivals.append(arr)

        if outlier_arrivals:
            logger.info(f"MAD filter: {len(outlier_arrivals)} outlier(s) flagged")

        return outlier_arrivals

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
    ########################
    # hypo7 classification
    #########################
    try:
        Q, QS, QD, classif_txt = classify_event(event, debug=True)
    except Exception as e:
        logger.error(f"\tError in classify_event: {e}")
        Q = 0
        QS = 0
        QD = 0
        classif_txt = "unknown"
    print(f"\thypo7 quality: {Q} ({classif_txt}), QS={QS}, QD={QD}")
    
    ########################
    # michele mod2 classification
    ########################
    try:
        mlq = classify_event_michele_mod2(event)
    except Exception as e:
        logger.error(f"\tError in classify_event_michele_mod2: {e}")
        mlq = ("N/A", "N/A")
    qf_str = f"{mlq[0]:.2f}" if mlq[0] is not None else "N/A"
    print(f"\tMichele mod2 quality: Q={mlq[1]}, QF={qf_str}")
    
    #########################
    # Gallacher GT5
    #########################
    origin = event.preferred_origin() if event.preferred_origin() else event.origins[0]
    try:
        gallacher_gt5  = compute_gallacher_gt5_score_obspy(origin)
    except Exception as e:
        logger.error(f"\tError in compute_gallacher_gt5_score_obspy: {e}")
        gallacher_gt5 = ("N/A", "N/A")
    print(f"\nGallacher GT5 score: {gallacher_gt5[0]}")
    pprint(gallacher_gt5[1])
    
    ########################
    # Table display
    ########################
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
