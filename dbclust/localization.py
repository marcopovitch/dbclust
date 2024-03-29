#!/usr/bin/env python
import sys
import io
import os
from math import isclose, fabs
import logging
import glob
import subprocess
import shlex
import tempfile
import numpy as np
import pandas as pd

import concurrent.futures
from functools import partial
import multiprocessing

import dask
import dask.bag as db
from dask import delayed

from itertools import combinations
import urllib.request
import urllib.parse
import copy

# from tqdm import tqdm
import dateparser
from obspy.core import UTCDateTime
from obspy import Catalog, read_events
from obspy.core.event import Origin, OriginQuality, ResourceIdentifier
from obspy import read_events
from jinja2 import Template

try:
    from quakeml import remove_duplicated_picks
except:
    from dbclust.quakeml import remove_duplicated_picks

# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("localization")
logger.setLevel(logging.INFO)


class NllLoc(object):
    def __init__(
        self,
        nll_bin,
        scat2latlon_bin,
        nll_times_path,
        nll_template,
        nll_obs_file=None,
        nll_channel_hint=None,
        nll_min_phase=4,
        nll_verbose=False,
        tmpdir="/tmp",
        min_station_with_P_and_S=0,
        double_pass=False,
        force_uncertainty=False,
        P_uncertainty=0.1,
        S_uncertainty=0.2,
        dist_km_cutoff=None,
        use_deactivated_arrivals=False,
        P_time_residual_threshold=None,
        S_time_residual_threshold=None,
        quakeml_settings=None,
        keep_scat=False,
        scat_file=None,
        log_level=logging.INFO,
    ):
        logger.setLevel(log_level)

        # define locator
        self.nll_bin = nll_bin
        self.scat2latlon_bin = scat2latlon_bin
        self.nll_time_path = nll_times_path
        self.nll_template = nll_template
        self.nll_obs_file = nll_obs_file  # obs file to localize
        self.nll_channel_hint = nll_channel_hint
        self.nll_min_phase = nll_min_phase
        self.nll_verbose = nll_verbose
        self.tmpdir = tmpdir
        self.min_station_with_P_and_S = min_station_with_P_and_S
        self.double_pass = double_pass
        self.force_uncertainty = force_uncertainty
        self.P_uncertainty = P_uncertainty
        self.S_uncertainty = S_uncertainty
        self.dist_km_cutoff = dist_km_cutoff
        self.use_deactivated_arrivals = use_deactivated_arrivals
        self.P_time_residual_threshold = P_time_residual_threshold
        self.S_time_residual_threshold = S_time_residual_threshold
        self.quakeml_settings = quakeml_settings
        self.keep_scat = keep_scat
        self.scat_file = scat_file

        # keep track of cluster affiliation
        self.event_cluster_mapping = {}

        # localization only if is nll_obs_file provided at init level
        if self.nll_obs_file:
            self.catalog = self.nll_localisation(
                nll_obs_file, double_pass=self.double_pass
            )
        else:
            self.catalog = Catalog()
        self.nb_events = len(self.catalog)

    @staticmethod
    def check_stations_with_P_and_S(event, origin, min_count):
        """
        Ensures that the number of stations with both P and S phases (count)
        is greater than or equal to the threshold (min_count).

        Returns count
        """
        count = {}
        for arrival in origin.arrivals:
            if hasattr(arrival, "time_weight") and arrival.time_weight == 0:
                continue
            pick = next(
                (p for p in event.picks if p.resource_id == arrival.pick_id), None
            )
            if not pick:
                continue
            wfid = pick.waveform_id
            station_name = f"{wfid.network_code}.{wfid.station_code}"
            phase_name = pick.phase_hint

            if station_name in count.keys():
                count[station_name].append(phase_name)
            else:
                count[station_name] = [phase_name]

        count = [len(count[k]) for k in count.keys()]
        return np.array([np.count_nonzero(x >= min_count) for x in count]).sum()

    def reloc_event(self, event):
        """
        Event relocalisation using a locator
        Returns a Catalog()
        """

        myevent = copy.deepcopy(event)

        show_event(myevent, "****", header=True)
        orig = myevent.preferred_origin()
        channel_hint = io.StringIO()
        for arrival in orig.arrivals:
            # res_id = ResourceIdentifier(prefix="smi:local")
            # arrival.resource_id = res_id
            pick = next(
                (p for p in myevent.picks if p.resource_id == arrival.pick_id), None
            )
            # keep channel info in channel_hint
            wfid = pick.waveform_id
            string = f"{wfid.network_code}_{wfid.station_code}_{wfid.location_code}_{wfid.channel_code}\n"
            channel_hint.write(string)

            if self.force_uncertainty:
                if "P" in pick.phase_hint or "p" in pick.phase_hint:
                    pick.time_errors.uncertainty = self.P_uncertainty
                elif "S" in pick.phase_hint or "s" in pick.phase_hint:
                    pick.time_errors.uncertainty = self.S_uncertainty

            # do not use pick with desactivated arrival
            if self.use_deactivated_arrivals == False and arrival.time_weight == 0:
                myevent.picks.remove(pick)
            elif (self.dist_km_cutoff is not None) and (
                arrival.distance > self.dist_km_cutoff / 111.0
            ):
                myevent.picks.remove(pick)

        channel_hint.seek(0)

        self.nll_obs_file = os.path.join(self.tmpdir, "nll_obs.txt")
        logger.debug(
            f"Writing nll_obs file to {self.nll_obs_file} in {self.tmpdir} directory."
        )
        myevent.write(self.nll_obs_file, format="NLLOC_OBS")
        self.nll_channel_hint = channel_hint
        cat = self.nll_localisation()
        # Fixme: add previously removed picks

        # add previous origin back to this event
        if cat:
            cat.events[0].origins.append(orig)
        else:
            logger.warning("relocation failed")
            logger.warning("fix me: should returns original location")

        return cat

    def nll_localisation(
        self,
        nll_obs_file=None,
        double_pass=None,
        pass_count=0,
        force_model_id=None,
        force_template=None,
    ):
        """
        Do the NLL stuff to localize event phases in nll_obs_file

        When double_pass is True, the localization is computed twice
        with picks/phases clean_up step.
        pass_count keeps track how many time relocation was done (do not modify this).

        Returns a multi-origin event in a Catalog()
        """
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
                "model_id" in self.quakeml_settings
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
        if os.path.exists(vel_file):
            with open(vel_file) as vel:
                model_id = vel.readline().strip()
                nll_template = vel.readline().strip()
                preloc_time = UTCDateTime(dateparser.parse(vel.readline().strip()))
                preloc_lat = float(vel.readline().strip())
                preloc_lon = float(vel.readline().strip())
                preloc_depth_m = float(vel.readline().strip())
            logger.info(
                f"Force localization to use model_id: {model_id}, template: {nll_template}."
            )
            if pass_count == 0:
                preloc_origin = Origin()
                preloc_origin.evaluation_mode = "automatic"
                preloc_origin.evaluation_status = "preliminary"
                preloc_origin.method_id = ResourceIdentifier("pyocto")
                preloc_origin.earth_model_id = ResourceIdentifier("haslach")
                if "agency_id" in self.quakeml_settings:
                    preloc_origin.agency_id = self.quakeml_settings["agency_id"]
                else:
                    preloc_origin.agency_id = "MyAgencyId"
                preloc_origin.latitude = preloc_lat
                preloc_origin.longitude = preloc_lon
                preloc_origin.depth = preloc_depth_m  # in meters
                preloc_origin.time = preloc_time
                preloc_origin.quality = OriginQuality()
                preloc_origin.quality.azimuthal_gap = 0

        logger.debug(f"Localization of {nll_obs_file} using {nll_template} template.")
        nll_obs_file_basename = os.path.basename(nll_obs_file)

        tmp_path = tempfile.mkdtemp(dir=self.tmpdir)
        conf_file = os.path.join(tmp_path, f"{nll_obs_file_basename}.conf")

        # path + root filename
        output = os.path.join(tmp_path, nll_obs_file_basename)

        # Values to be substituted in the template
        tags = {
            "OBSFILE": nll_obs_file,
            "NLL_TIME_PATH": self.nll_time_path,
            "OUTPUT": output,
            "NLL_MIN_PHASE": self.nll_min_phase,
        }

        # Generate NLL configuration file
        try:
            self.replace(nll_template, conf_file, tags)
        except Exception as e:
            logger.error(e)
            return Catalog()

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

        # check if location was rejected

        # check from stdout if there is any missing station grid file
        # WARNING: cannot open grid buffer file: nll_times/auvergne-pyrocko2/auvergne.P.01x02.time.buf
        for line in result.stdout.splitlines():
            if "WARNING: cannot open grid buffer file" in line:
                logger.error(line)
            elif any(k in line for k in ("ABORTED", "IGNORED", "REJECTED")):
                why = " ".join(line.split()[3:]).replace('"', "")
                logger.info(f"Localization was ABORTED|IGNORED|REJECTED: {why}")
                return Catalog()

        if self.nll_verbose:
            print(result.stdout)

        # Read results
        nll_output = os.path.join(tmp_path, "last.hyp")
        try:
            cat = read_events(nll_output)
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

        # check for nan value in uncertainty
        if "nan" in [
            str(o.latitude_errors.uncertainty),
            str(o.longitude_errors.uncertainty),
            str(o.depth_errors.uncertainty),
        ]:
            logger.debug("Found NaN value in uncertainty. Ignoring event !")
            return Catalog()

        # nll_channel_hint allows to keep track of net, sta, loc, chan values
        # and could be None, a file or a StringIO
        if self.nll_channel_hint:
            if isinstance(self.nll_channel_hint, io.StringIO):
                self.nll_channel_hint.seek(0)
                logger.debug("nll_channel_hint uses StringIO()")
            else:
                logger.debug(f"nll_channel_hint use {self.nll_channel_hint}")
            cat = self.fix_wfid(cat, self.nll_channel_hint)
        else:
            logger.warning("No nll_channel_hint file provided !")

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

        if self.force_uncertainty:
            for pick in e.picks:
                if "P" in pick.phase_hint or "p" in pick.phase_hint:
                    pick.time_errors.uncertainty = self.P_uncertainty
                elif "S" in pick.phase_hint or "s" in pick.phase_hint:
                    pick.time_errors.uncertainty = self.S_uncertainty

        # add preloc origin to event (only basic info) # only the first pass
        if preloc_origin:
            e.origins.append(preloc_origin)

        # try a relocation
        if self.double_pass and pass_count == 0:
            logger.debug("Starting double pass relocation.")
            cat2 = cat.copy()
            event2 = cat2.events[0]
            event2 = self.cleanup_pick_phase(event2)
            if len(event2.picks):
                new_nll_obs_file = nll_obs_file + ".2nd_pass"
                cat2.write(new_nll_obs_file, format="NLLOC_OBS")
                cat2 = self.nll_localisation(
                    new_nll_obs_file,
                    double_pass=self.double_pass,
                    pass_count=1,
                    force_model_id=model_id,
                    force_template=nll_template,
                )
            else:
                cat2 = None

            if cat2:
                event2 = cat2.events[0]
                orig2 = event2.preferred_origin()
                # add this new origin to catalog and set it as preferred
                e.origins.append(orig2)
                e.preferred_origin_id = orig2.resource_id
                e.picks += event2.picks
                e = remove_duplicated_picks(e)
            else:
                # can't relocate: set it to "not existing"
                e.event_type = "not existing"

        # if there is only one origin, set it to the preferred
        if len(e.origins) == 1:
            e.preferred_origin_id = e.origins[0].resource_id

        return cat

    def processes_get_localisations_from_nllobs_dir(self, OBS_PATH, append=True):
        obs_files_pattern = os.path.join(OBS_PATH, "cluster-*.obs")
        logger.debug(f"Localization of {obs_files_pattern}")

        my_loc_proc = partial(self.nll_localisation, double_pass=self.double_pass)
        processes = []

        for f in glob.glob(obs_files_pattern):
            process = multiprocessing.Process(target=my_loc_proc, args=(f,))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        cat_results = [process.exitcode for process in processes]

        mycatalog = Catalog()
        for cat in cat_results:
            if not cat:
                continue
            # there is always only one event in the catalog
            e = cat.events[0]
            o = e.preferred_origin()
            # nb_station_used = o.quality.used_station_count
            o.quality.used_station_count = self.get_used_station_count(e, o)
            # nb_station_used = o.quality.used_station_count
            nb_phase_used = o.quality.used_phase_count
            # if nb_station_used >= self.nll_min_phase:
            if nb_phase_used >= self.nll_min_phase:
                count = self.check_stations_with_P_and_S(
                    e, o, self.min_station_with_P_and_S
                )
                if count >= self.min_station_with_P_and_S:
                    logger.info(
                        f"{nb_phase_used} phases, {o.quality.used_station_count} stations "
                        f"and {count} (min: {self.min_station_with_P_and_S}) stations with P and S (both)."
                    )
                    mycatalog += cat
                else:
                    logger.info(
                        f"Not enough stations with both P and S ({count}, min:{self.min_station_with_P_and_S})"
                        "... ignoring it !"
                    )
            else:
                logger.debug(
                    f"Not enough phases ({nb_phase_used}/{self.nll_min_phase}) for event"
                    f" ... ignoring it !"
                )

        if append is True:
            self.catalog += mycatalog

        # sort events by time
        self.nb_events = len(self.catalog)
        if self.nb_events > 1:
            self.catalog.events = sorted(
                self.catalog.events, key=lambda e: e.preferred_origin().time
            )
        return mycatalog

    def multiproc_get_localisations_from_nllobs_dir(self, OBS_PATH, append=True):
        obs_files_pattern = os.path.join(OBS_PATH, "cluster-*.obs")
        logger.debug(f"Localization of {obs_files_pattern}")

        max_workers = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=max_workers)

        my_loc_proc = partial(self.nll_localisation, double_pass=self.double_pass)
        cat_results = pool.map(my_loc_proc, glob.glob(obs_files_pattern))

        pool.close()
        pool.join()

        mycatalog = Catalog()
        for cat in cat_results:
            if not cat:
                continue
            # there is always only one event in the catalog
            e = cat.events[0]
            o = e.preferred_origin()
            # nb_station_used = o.quality.used_station_count
            o.quality.used_station_count = self.get_used_station_count(e, o)
            # nb_station_used = o.quality.used_station_count
            nb_phase_used = o.quality.used_phase_count
            # if nb_station_used >= self.nll_min_phase:
            if nb_phase_used >= self.nll_min_phase:
                count = self.check_stations_with_P_and_S(
                    e, o, self.min_station_with_P_and_S
                )
                if count >= self.min_station_with_P_and_S:
                    logger.info(
                        f"{nb_phase_used} phases, {o.quality.used_station_count} stations "
                        f"and {count} (min: {self.min_station_with_P_and_S}) stations with P and S (both)."
                    )
                    mycatalog += cat
                else:
                    logger.info(
                        f"Not enough stations with both P and S ({count}, min:{self.min_station_with_P_and_S})"
                        "... ignoring it !"
                    )
            else:
                logger.debug(
                    f"Not enough phases ({nb_phase_used}/{self.nll_min_phase}) for event"
                    f" ... ignoring it !"
                )

        if append is True:
            self.catalog += mycatalog

        # sort events by time
        self.nb_events = len(self.catalog)
        if self.nb_events > 1:
            self.catalog.events = sorted(
                self.catalog.events, key=lambda e: e.preferred_origin().time
            )
        return mycatalog

    def thread_get_localisations_from_nllobs_dir(self, OBS_PATH, append=True):
        obs_files_pattern = os.path.join(OBS_PATH, "cluster-*.obs")
        logger.debug(f"Localization of {obs_files_pattern}")

        max_workers = multiprocessing.cpu_count()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            my_loc_proc = partial(self.nll_localisation, double_pass=self.double_pass)
            cat_results = list(executor.map(my_loc_proc, glob.glob(obs_files_pattern)))

        mycatalog = Catalog()
        for cat in cat_results:
            if not cat:
                continue
            # there is always only one event in the catalog
            e = cat.events[0]
            o = e.preferred_origin()
            # nb_station_used = o.quality.used_station_count
            o.quality.used_station_count = self.get_used_station_count(e, o)
            # nb_station_used = o.quality.used_station_count
            nb_phase_used = o.quality.used_phase_count
            # if nb_station_used >= self.nll_min_phase:
            if nb_phase_used >= self.nll_min_phase:
                count = self.check_stations_with_P_and_S(
                    e, o, self.min_station_with_P_and_S
                )
                if count >= self.min_station_with_P_and_S:
                    logger.info(
                        f"{nb_phase_used} phases, {o.quality.used_station_count} stations "
                        f"and {count} (min: {self.min_station_with_P_and_S}) stations with P and S (both)."
                    )
                    mycatalog += cat
                else:
                    logger.info(
                        f"Not enough stations with both P and S ({count}, min:{self.min_station_with_P_and_S})"
                        "... ignoring it !"
                    )
            else:
                logger.debug(
                    f"Not enough phases ({nb_phase_used}/{self.nll_min_phase}) for event"
                    f" ... ignoring it !"
                )

        if append is True:
            self.catalog += mycatalog

        # sort events by time
        self.nb_events = len(self.catalog)
        if self.nb_events > 1:
            self.catalog.events = sorted(
                self.catalog.events, key=lambda e: e.preferred_origin().time
            )
        return mycatalog

    def dask_bag_get_localisations_from_nllobs_dir(self, OBS_PATH, append=True):
        """
        nll localisation and export to quakeml
        warning : network and channel are lost since they are not used by nll
        use Phase() to get them.
        if append is True, the obtain catalog is appended to the NllLoc catalog

        returns a catalog
        """
        obs_files_pattern = os.path.join(OBS_PATH, "cluster-*.obs")
        logger.debug(f"Localization of {obs_files_pattern}")

        b = db.from_sequence(
            glob.glob(obs_files_pattern), partition_size=multiprocessing.cpu_count()
        )
        cat_results = b.map(
            lambda x: self.nll_localisation(x, double_pass=self.double_pass)
        ).compute()

        mycatalog = Catalog()
        for cat in cat_results:
            if not cat:
                continue
            # there is always only one event in the catalog
            e = cat.events[0]
            o = e.preferred_origin()
            # nb_station_used = o.quality.used_station_count
            o.quality.used_station_count = self.get_used_station_count(e, o)
            # nb_station_used = o.quality.used_station_count
            nb_phase_used = o.quality.used_phase_count
            # if nb_station_used >= self.nll_min_phase:
            if nb_phase_used >= self.nll_min_phase:
                count = self.check_stations_with_P_and_S(
                    e, o, self.min_station_with_P_and_S
                )
                if count >= self.min_station_with_P_and_S:
                    logger.info(
                        f"{nb_phase_used} phases, {o.quality.used_station_count} stations "
                        f"and {count} (min: {self.min_station_with_P_and_S}) stations with P and S (both)."
                    )
                    mycatalog += cat
                else:
                    logger.info(
                        f"Not enough stations with both P and S ({count}, min:{self.min_station_with_P_and_S})"
                        "... ignoring it !"
                    )
            else:
                logger.debug(
                    f"Not enough phases ({nb_phase_used}/{self.nll_min_phase}) for event"
                    f" ... ignoring it !"
                )

        if append is True:
            self.catalog += mycatalog

        # sort events by time
        self.nb_events = len(self.catalog)
        if self.nb_events > 1:
            self.catalog.events = sorted(
                self.catalog.events, key=lambda e: e.preferred_origin().time
            )
        return mycatalog

    def dask_get_localisations_from_nllobs_dir(self, OBS_PATH, append=True):
        """
        nll localisation and export to quakeml
        warning : network and channel are lost since they are not used by nll
        use Phase() to get them.
        if append is True, the obtain catalog is appended to the NllLoc catalog

        returns a catalog
        """
        obs_files_pattern = os.path.join(OBS_PATH, "cluster-*.obs")
        logger.debug(f"Localization of {obs_files_pattern}")

        delayed_tasks = [
            dask.delayed(self.nll_localisation)(f, double_pass=self.double_pass)
            for f in glob.glob(obs_files_pattern)
        ]
        # execute tasks
        cat_results = dask.compute(*delayed_tasks)

        mycatalog = Catalog()
        for cat in cat_results:
            if not cat:
                continue
            # there is always only one event in the catalog
            e = cat.events[0]
            o = e.preferred_origin()
            # nb_station_used = o.quality.used_station_count
            o.quality.used_station_count = self.get_used_station_count(e, o)
            # nb_station_used = o.quality.used_station_count
            nb_phase_used = o.quality.used_phase_count
            # if nb_station_used >= self.nll_min_phase:
            if nb_phase_used >= self.nll_min_phase:
                count = self.check_stations_with_P_and_S(
                    e, o, self.min_station_with_P_and_S
                )
                if count >= self.min_station_with_P_and_S:
                    logger.info(
                        f"{nb_phase_used} phases, {o.quality.used_station_count} stations "
                        f"and {count} (min: {self.min_station_with_P_and_S}) stations with P and S (both)."
                    )
                    mycatalog += cat
                else:
                    logger.info(
                        f"Not enough stations with both P and S ({count}, min:{self.min_station_with_P_and_S})"
                        "... ignoring it !"
                    )
            else:
                logger.debug(
                    f"Not enough phases ({nb_phase_used}/{self.nll_min_phase}) for event"
                    f" ... ignoring it !"
                )

        if append is True:
            self.catalog += mycatalog

        # sort events by time
        self.nb_events = len(self.catalog)
        if self.nb_events > 1:
            self.catalog.events = sorted(
                self.catalog.events, key=lambda e: e.preferred_origin().time
            )
        return mycatalog

    def get_localisations_from_nllobs_dir(self, OBS_PATH, append=True):
        """nll localisation and export to quakeml

        warning : network and channel are lost since they are not used by nll
        use Phase() to get them back.

        Args:
            OBS_PATH (string): directory where are the nll obs files
            append (bool, optional): append new origin to self.catalog. Defaults to True.

        Returns:
            Catalog: returns a catalog of all computed origins
        """
        obs_files_pattern = os.path.join(OBS_PATH, "cluster-*.obs")
        logger.debug(f"Localization of {obs_files_pattern}")

        mycatalog = Catalog()
        for nll_obs_file in sorted(glob.glob(obs_files_pattern)):
            # localization
            cat = self.nll_localisation(nll_obs_file, double_pass=self.double_pass)
            if not cat:
                logger.debug(f"No loc obtained for {nll_obs_file}:/")
                continue

            # there is always only one event in the catalog
            e = cat.events[0]
            o = e.preferred_origin()
            o.quality.used_station_count = self.get_used_station_count(e, o)
            # nb_station_used = o.quality.used_station_count
            nb_phase_used = o.quality.used_phase_count
            # if nb_station_used >= self.nll_min_phase:
            if nb_phase_used >= self.nll_min_phase:
                count = self.check_stations_with_P_and_S(
                    e, o, self.min_station_with_P_and_S
                )
                if count >= self.min_station_with_P_and_S:
                    logger.info(
                        f"{nb_phase_used} phases, {o.quality.used_station_count} stations "
                        f"and {count} (min: {self.min_station_with_P_and_S}) stations with P and S (both)."
                    )
                    mycatalog += cat
                else:
                    logger.info(
                        f"Not enough stations with both P and S ({count}, min:{self.min_station_with_P_and_S})"
                        "... ignoring it !"
                    )
            else:
                logger.debug(
                    f"Not enough phases ({nb_phase_used}/{self.nll_min_phase}) for event"
                    f" ... ignoring it !"
                )

        if append is True:
            self.catalog += mycatalog

        # sort events by time
        self.nb_events = len(self.catalog)
        if self.nb_events > 1:
            self.catalog.events = sorted(
                self.catalog.events, key=lambda e: e.preferred_origin().time
            )

        return mycatalog

    def cleanup_pick_phase(self, event):
        """
        Remove picks/arrivals with:
            - time weight set to 0
            - bad residual
            - duplicated phases (remove the one with highest residual)
        """
        orig = event.preferred_origin()
        pick_to_delete = []
        arrival_to_delete = []
        for arrival in orig.arrivals:
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
                isclose(arrival.time_weight, 0, abs_tol=0.001)
                or bad_time_residual
                or (
                    self.dist_km_cutoff is not None
                    and arrival.distance > self.dist_km_cutoff / 111.0
                )
            ):
                pick = next(
                    (p for p in event.picks if p.resource_id == arrival.pick_id), None
                )
                pick_to_delete.append(pick)
                arrival_to_delete.append(arrival)
                # logger.info(pick)
                # logger.info(arrival)

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
                    f"Duplicated pick detected [{p.waveform_id.get_seed_string()}, {a.phase}, {p.time}]... removing the one with highest residual"
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
        orig.quality.used_phase_count = len(
            [a.time_weight for a in orig.arrivals if a.time_weight]
        )
        orig.quality.used_station_count = NllLoc.get_used_station_count(event, orig)
        return event

    @staticmethod
    def get_used_station_count(event, origin):
        station_list = []
        # origin = event.preferred_origin()
        for arrival in origin.arrivals:
            if arrival.time_weight and arrival.time_residual:
                pick = next(
                    (p for p in event.picks if p.resource_id == arrival.pick_id), None
                )
                if pick:
                    station_list.append(
                        f"{pick.waveform_id.network_code}.{pick.waveform_id.station_code}"
                    )
        return len(set(station_list))

    @staticmethod
    def get_used_phase_count(event, origin):
        nb_phase_used = 0
        for arrival in origin.arrivals:
            if arrival.time_weight and arrival.time_residual:
                pick = next(
                    (p for p in event.picks if p.resource_id == arrival.pick_id), None
                )
                if pick:
                    nb_phase_used += 1
        return nb_phase_used

    @staticmethod
    def replace(templatefile, outfilename, tags):
        with open(templatefile) as file_:
            template = Template(file_.read())
        t = template.render(tags)
        with open(outfilename, "w") as out_fh:
            out_fh.write(t)
            logger.debug(f"Template {templatefile} rendered as {outfilename}")

    @staticmethod
    def read_chan(fname):
        df = pd.read_csv(
            fname,
            sep="_",
            names=["net", "sta", "loc", "chan"],
            header=None,
            dtype=object,
        )
        df["chan"] = df["chan"].str[:-1]
        df = df.drop_duplicates().fillna("")
        return df

    def fix_wfid(self, cat, wfid_hint):
        """
        wfid_hint is a file with (net, sta, loc, chan) information
        to be used as hint to populate quakeml from nll obs file
        """

        msi_cmd = str(
            "wfid_hint is a file with (net, sta, loc, chan) information\n"
            "to be used as a hint to populate quakeml from nll obs file.\n"
        )

        try:
            df = self.read_chan(wfid_hint)
        except Exception as e:
            logger.error(
                f"Something went wrong with {wfid_hint} file ... Nothing was done !"
            )
            logger.error(e)
            logger.error(msi_cmd)
            return cat

        for e in cat.events:
            for p in e.picks:
                # use inventory or ws-event
                # wfid = get_station_wfid()

                sta = p.waveform_id.station_code

                if p.waveform_id.network_code == "":
                    net = df[df["sta"] == sta]["net"].drop_duplicates()
                    if len(net) == 0:
                        logger.warning(
                            f"Network code not found for station {sta} (filtered ?)"
                        )
                        continue
                    elif len(net) != 1:
                        if logger.level == logging.DEBUG:
                            logger.warning(
                                f"[wfid_hint] Duplicated network code for station {sta}"
                            )
                            logger.warning(f"    using the first one {net.iloc[0]}")
                    net = net.iloc[0]
                    p.waveform_id.network_code = net

                if (
                    p.waveform_id.location_code is None
                    or p.waveform_id.location_code == ""
                ):
                    loc = df[(df["sta"] == sta) & (df["net"] == net)][
                        "loc"
                    ].drop_duplicates()
                    if len(loc) == 0:
                        logger.warning("Location code not found for {net}.{sta}")
                    if len(loc) != 1:
                        if logger.level == logging.DEBUG:
                            logger.warning(
                                f"[wfid_hint] Duplicated location code for {net}.{sta}"
                            )
                            logger.warning(f"    using the first one {loc.iloc[0]}")
                    loc = loc.iloc[0]
                    p.waveform_id.location_code = loc

                chan = df[(df["sta"] == sta) & (df["net"] == net) & (df["loc"] == loc)][
                    "chan"
                ].drop_duplicates()
                if len(chan) == 0:
                    logger.warning("Channel code not found for {net}.{sta}")
                elif len(chan) != 1:
                    if logger.level == logging.DEBUG:
                        logger.warning(
                            f"[wfid_hint] Duplicated channel code for {net}.{sta}.{loc}"
                        )
                        logger.warning(f"    using the first one {chan.iloc[0]}")
                chan = chan.iloc[0]

                if "P" in p.phase_hint or "p" in p.phase_hint:
                    p.waveform_id.channel_code = f"{chan}Z"

                if "S" in p.phase_hint or "s" in p.phase_hint:
                    p.waveform_id.channel_code = f"{chan}N"
        return cat

    def show_localizations(self):
        print("%d events in catalog:" % len(self.catalog))
        print("Text, T0, lat, lon, depth(m), RMS, sta_count, phase_count, gap1, gap2")
        for e in self.catalog.events:
            try:
                nll_obs = self.event_cluster_mapping[e.resource_id.id]
            except:
                nll_obs = ""
            show_event(e, nll_obs)


def get_pick_from_arrival(event, arrival):
    pick = next((p for p in event.picks if p.resource_id == arrival.pick_id), None)
    return pick


def show_event(event, txt="", header=False):
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


def show_origin(o, txt):
    if o.quality.azimuthal_gap:
        azimuthal_gap = f"{o.quality.azimuthal_gap:.1f}"
    else:
        # logger.warning("No azimuthal_gap defined !")
        azimuthal_gap = "-"

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
                    o.quality.standard_error,
                    o.quality.used_station_count,
                    o.quality.used_phase_count,
                    azimuthal_gap,
                    f"{o.quality.secondary_azimuthal_gap:.1f}"
                    if o.quality.secondary_azimuthal_gap
                    else "-",
                    o.earth_model_id.id.split("/")[-1],
                    o.method_id.id.split("/")[-1],
                ],
            )
        )
    )


def reloc_fdsn_event(locator, eventid, fdsnws):
    link = f"{fdsnws}/query?eventid={urllib.parse.quote(eventid, safe='')}&includearrivals=true"
    logger.debug(link)

    try:
        with urllib.request.urlopen(link) as f:
            cat = read_events(f.read())
    except Exception as e:
        logger.error("Error getting/reading eventid %s", eventid)
        sys.exit()

    if not cat:
        logger.error("[%s] no such eventid or no origin !", eventid)
        sys.exit()

    event = cat[0]
    cat = locator.reloc_event(event)
    return cat


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)

    nlloc_bin = "NLLoc"
    scat2latlon_bin = "scat2latlon"
    nlloc_times_path = "/Users/marc/Dockers/routine/nll/data/times"
    nlloc_template = "../nll_template/nll_haslach_template.conf"

    # eventid = "smi:local/437618f7-9cfe-4616-8e23-fdf32f7155db"
    # fdsnws = "http://localhost:10003/fdsnws/event/1"
    # nlloc_template = "../nll_template/nll_auvergne_template.conf"

    fdsnws = "https://api.franceseisme.fr/fdsnws/event/1"
    eventid = "fr2023lfhbcx"

    force_uncertainty = True
    P_uncertainty = 0.05
    S_uncertainty = 0.1

    # nlloc_bin,
    # nlloc_times_path,
    # nlloc_template,
    # nll_obs_file=None,
    # nll_channel_hint=None,
    # tmpdir="/tmp",
    # double_pass=False,
    # force_uncertainty=False,
    # P_uncertainty=0.1,
    # S_uncertainty=0.2,
    # dist_km_cutoff = None,
    # P_time_residual_threshold=None,
    # S_time_residual_threshold=None,
    # nll_min_phase=4,
    # verbose=False,

    with tempfile.TemporaryDirectory() as tmpdir:
        locator = NllLoc(
            nlloc_bin,
            scat2latlon_bin,
            nlloc_times_path,
            nlloc_template,
            #
            # nll_channel_hint=channel_hint,
            # nll_obs_file=obs.name,
            tmpdir=tmpdir,
            #
            force_uncertainty=force_uncertainty,
            P_uncertainty=P_uncertainty,
            S_uncertainty=S_uncertainty,
            # dist_km_cutoff=None,  # KM
            #
            double_pass=True,
            # P_time_residual_threshold=0.45,
            # S_time_residual_threshold=0.75,
            #
            nll_verbose=True,
        )

        cat = reloc_fdsn_event(locator, eventid, fdsnws)
        for e in cat:
            show_event(e, "****", header=True)

        # cat.write(f"{urllib.parse.quote(eventid, safe='')}.qml", format="QUAKEML")
        # cat.write(f"{urllib.parse.quote(eventid, safe='')}.sc3ml", format="SC3ML")
