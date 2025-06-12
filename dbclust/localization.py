#!/usr/bin/env python
# -*- coding: utf-8 -*-
import concurrent.futures
import copy
import glob
import io
import json
import logging
import multiprocessing
import os
import re
import shlex
import subprocess
import sys
import tempfile
import traceback
import urllib.parse
import urllib.request
import warnings
from collections import defaultdict
from functools import partial
from itertools import combinations
from math import fabs
from math import isclose
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import dateparser
import geopandas as gpd
import numpy as np
import pandas as pd
import ray
from icecream import ic
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
from ray.util.multiprocessing import Pool
from shapely import distance
from shapely import prepare
from shapely import within
from shapely.geometry import Point

from dbclust.config import Zone
from dbclust.config import Zones
from dbclust.gap import compute_azimuthal_gap
from dbclust.gap import compute_gap
from dbclust.gap import compute_secondary_azimuthal_gap
from dbclust.gap import get_arrival_with_distance_gap_greater_than
from dbclust.localization_quality import classify_event
from dbclust.plot import plot_arrival_time
from dbclust.quakeml import deduplicate_picks
from dbclust.relabel import get_best_polygon_for_point
from dbclust.relabel import relabel_phase_and_comment_arrival
#import dask
#import dask.bag as db

# Disable warnings from obspy
# UserWarning: Setting attribute ... which is not a default attribute
warnings.filterwarnings("ignore", category=UserWarning, module="obspy")

# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("localization")
logger.setLevel(logging.INFO)


# Define the preferred phase order
phase_order = ["Pg", "Sg", "Pn", "Sn", "P", "S"]

# set time_weight tolerance
time_weight_tolerance = 0.01


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
        log_level=logging.INFO,
    ):
        logger.setLevel(log_level)

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
                continue  # Ignore les arrivals avec un poids nul
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
        except LocalizationError as e:
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during localization: {e}")
            traceback.print_exc()
            raise e

        # Add the previous event or origin back to this event.
        if cat:
            new_loc = cat.events[0]
            new_loc.origins.append(orig)
            new_loc.picks.extend(event.picks)
        else:
            raise LocalizationError(f"using {self.loc_method} method.")

        return cat

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

        if double_pass != None:
            # force double pass
            self.double_pass = double_pass
        else:
            # use the value defined in locator
            pass

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
        # get velocity model to use thanks to preliminary location
        if os.path.exists(vel_file):
            with open(vel_file) as vel:
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
        except Exception as e:
            ic(nll_template, conf_file, tags)
            raise e

        ####################
        # NLL Localization #
        ####################
        cmde = f"{self.nll_bin} {conf_file}"
        logger.debug(cmde)

        try:
            result = subprocess.run(
                shlex.split(cmde),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(e)
            return Catalog()
        except Exception as e:
            logger.error(e)
            return Catalog()

        if result.returncode != 0:
            logger.error(
                f"!!! Something went wrong using: {cmde}, "
                f"returned code is {result.returncode}\n"
                f"{result.stdout}"
            )
            for p in picks:
                logger.error(p)
            return Catalog()

        # check from stdout if there is any missing station grid file
        for line in result.stdout.splitlines():
            if "WARNING: cannot open grid buffer file" in line:
                logger.error(line)
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
                logger.warning(
                    f"Station {l[2]} outside velocity bounding box coordinates: localization aborted by NonLinLoc !"
                )
                if self.nll_verbose:
                    print(result.stdout)
                return Catalog()
            elif "ERROR: calc_maximum_likelihood_ot:" in line:
                # localization failed. It appends when using EDT_OT_WT
                # raise an exception to try relocation with another method
                loc_method_used = (
                    self.loc_method if force_loc_method is None else force_loc_method
                )
                raise LocalizationError(f"using {self.loc_method} method.")
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
                if len(l) > 1:
                    expect_lat = float(l[2])
                    expect_lon = float(l[4])
                    expect_depth = float(l[6])

        if self.nll_verbose:
            print(result.stdout)

        # Read results
        nll_output = os.path.join(tmp_path, "last.hyp")
        try:
            # use picks to map picks information
            cat = read_events(nll_output, picks=picks)
        except Exception as e:
            # No localization
            logger.debug(e)
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
                result = subprocess.run(
                    shlex.split(cmde),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                logger.error(e)
            except Exception as e:
                logger.error(e)

            self.scat_file = os.path.join(tmp_path, "last.hyp.scat.xyz")
            logger.debug("nll scat file is %s", self.scat_file)

        # there is always only one event in the catalog
        # fixme: use resource_id to forge *better* eventid and originid
        e = cat.events[0]
        o = e.preferred_origin()
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
        o.comments.append(Comment(text='{"scatter_volume": %s}' % (scatter_volume)))

        # store into comment expectation hypocenter
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

        # try a relocation
        if self.double_pass and pass_count == 0:
            logger.debug("Starting double pass relocation.")
            cat2 = cat.copy()
            event2 = cat2.events[0]
            # event2 = deduplicate_picks(event2)

            # unset arrival with gap in distance > dist_max
            event2 = self.unset_arrival_gap_dist_km(event2, self.gap_dist_max_km)

            # Clean up picks outside of the polygons defined in zones
            if self.use_pick_zone and self.zones and self.enable_cleanup_pick_zone:
                # To be done:
                # 1. remove picks/arrivals with time_weight set to 0
                # 2. remove picks/arrivals with duplicated phases
                # 3. remove picks/arrivals with distance > dist_km_cutoff
                # 4. relabel pick within zone
                if self.force_zone_name:
                    zone = self.zones.get_zone_from_name(self.force_zone_name)
                else:
                    zone, _ = self.zones.find_zone(o.latitude, o.longitude)

                # keep track of relabel for later user
                # as info on the event will be lost
                if len(zone.picks_delimiter):
                    event2, relabel_dict = self.cleanup_picks_and_relabel_picks(
                        event2, zone, eval_threshold=self.min_score_threshold_pick_zone
                    )
                else:
                    # no zone found: use default cleanup
                    event2 = self.cleanup_pick_phase(event2)
                    relabel_dict = {}
            else:
                # legacy code to clean up pick :
                # 1. with bad residual
                # 2. with time_weight set to 0
                # 3. with distance > dist_km_cutoff
                # 4. with duplicated phases
                event2 = self.cleanup_pick_phase(event2)
                relabel_dict = {}

            if len(event2.picks):
                new_nll_obs_file = nll_obs_file + ".2nd_pass"
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
                    logger.debug(f"{ex} - in double pass")
                    cat2 = None
                except Exception as ex:
                    logger.error(f"Unexpected error during localization: {ex}")
                    cat2 = None
            else:
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
                e.origins.append(orig2)
                e.preferred_origin_id = orig2.resource_id
                # e.picks += event2.picks
                # e = deduplicate_picks(e)
            else:
                # can't relocate: set it to "not existing"
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
            e.picks.extend(preloc_picks_list)
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
            assert pick, f"Can't find pick for arrival {a.pick_id}"

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

        logger.debug(
            f"cleanup: remove {len(arrival_to_delete)} phases and {len(pick_to_delete)} picks."
        )

        for a in arrival_to_delete:
            orig.arrivals.remove(a)
        for p in pick_to_delete:
            event.picks.remove(p)

        # check duplicated picks
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
                logger.info(
                    f"Duplicated pick detected [{p.waveform_id.get_seed_string()}, {a.phase}, {p.time}]... "
                    f"removing the one with highest residual"
                )
                if p not in pick_to_delete:
                    pick_to_delete.append(p)
                if a not in arrival_to_delete:
                    arrival_to_delete.append(a)

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
        with open(templatefile) as file_:
            template = Template(file_.read())
        t = template.render(tags)
        with open(outfilename, "w") as out_fh:
            out_fh.write(t)
            logger.debug(f"Template {templatefile} rendered as {outfilename}")

    def show_localizations(self) -> None:
        print("%d events in catalog:" % len(self.catalog))
        print("Text, T0, lat, lon, depth(m), RMS, sta_count, phase_count, gap1, gap2")
        for e in self.catalog.events:
            try:
                nll_obs = self.event_cluster_mapping[e.resource_id.id]
            except:
                nll_obs = ""
            show_event(e, nll_obs)


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


def show_event(event: Event, txt: str = "", header: bool = False):
    if header:
        print(
            "Text, T0, lat, lon, depth, RMS, sta_count, phase_count, gap1, gap2, model, locator"
        )

    o_pref = event.preferred_origin()

    if hasattr(event, "event_type") and event.event_type == "not existing":
        show_origin(o_pref, "FAKE")
    else:
        show_origin(o_pref, txt)

    for o in event.origins:
        if o == o_pref:
            continue
        show_origin(o, " |__")


def show_origin(o: Origin, txt: str) -> None:
    # if hasattr(o, "quality") and o.quality.azimuthal_gap:
    #     azimuthal_gap = f"{o.quality.azimuthal_gap:.1f}"
    # else:
    #     # logger.warning("No azimuthal_gap defined !")
    #     azimuthal_gap = "-"

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

    print(
        ", ".join(
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
    )


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
            except:
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
            except:
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
    try:
        Q, QS, QD, classif_txt = classify_event(event, debug=True)
    except Exception as e:
        logger.error(f"Error in classify_event: {e}")
        Q = 0
        QS = 0
        QD = 0
        classif_txt = "unknown"
    print(f"quality: {Q} ({classif_txt}), QS={QS}, QD={QD}")
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
            with urllib.request.urlopen(link) as f:
                cat = read_events(f.read())
        except Exception as e:
            raise ValueError(
                f"Error with {link}, cant't get/read eventid {eventid} ({e})"
            )

        if not cat:
            raise ValueError(f"[{eventid}] no such eventid !")

        event = cat[0]
    else:
        eventid = event.resource_id.id

    if locator.zones:
        if zone_name:
            logger.info(f"Forcing zone to {zone_name}.")
            zone = locator.zones.get_zone_from_name(zone_name)
        else:
            # Find the zone and set the velocity model and the nll template
            origin = event.preferred_origin() or event.origins[0]
            zone, _ = locator.zones.find_zone(origin.latitude, origin.longitude)
        if zone.empty:
            locator.zones.show_zones()
            raise ValueError(f"Zone {zone_name} not found.")

        locator.quakeml_settings["model_id"] = zone.velocity_profile
        logger.info(
            f"Using {zone['name']} zone, {locator.quakeml_settings['model_id']} model_id for event {eventid}."
        )

        # Set the nll template according to the zone
        if zone.empty:
            locator.zones.show_zones()
            raise ValueError(
                f'No template defined for zone {locator.quakeml_settings["model_id"]} !'
            )
        locator.nll_template = zone["template"]

    else:
        logger.warning(
            f'No zones defined, using default velocity model {locator.quakeml_settings["model_id"]}.'
        )

        # get the default template

    try:
        cat = locator.reloc_event(event)
    except LocalizationError as e:
        raise e
    except Exception as e:
        logger.exception(f"Unexpected localization error for event {eventid}: {e}")
        traceback.print_exc()
        raise e

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
    with open(o_parameters_file) as vel:
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
        agency_id=preloc_origin.agency_id,
        author=author,
        creation_time=UTCDateTime.now(),
    )
    preloc_origin.creation_info = CreationInfo(
        creation_time=UTCDateTime(),
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
        net, sta = row["station"].split(".")[:2]
        try:
            loc = row["station"].split(".")[2]
        except:
            loc = ""
        try:
            chan = row["station"].split(".")[3]
        except:
            chan = ""

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
