#!/usr/bin/env python
import sys
import io
import os
from math import isclose, fabs
import logging
import glob
import pathlib
import subprocess
import shlex
import tempfile
import pandas as pd
import multiprocessing
from dask.distributed import Client, LocalCluster
import dask.bag as db
from itertools import product, combinations

# from tqdm import tqdm
from obspy import Catalog, read_events
from obspy import read_events
from obspy.core.event import ResourceIdentifier
from jinja2 import Template

# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("localization")
logger.setLevel(logging.INFO)


class NllLoc(object):
    def __init__(
        self,
        nlloc_bin,
        nlloc_times_path,
        nlloc_template,
        nll_obs_file=None,
        nll_channel_hint=None,
        tmpdir="/tmp",
        double_pass=False,
        force_uncertainty=False,
        P_uncertainty=0.1,
        S_uncertainty=0.2,
        P_time_residual_threshold=None,
        S_time_residual_threshold=None,
        nll_min_phase=4,
        log_level=logging.INFO,
        nll_verbose=False,
    ):
        logger.setLevel(log_level)

        # define locator
        self.nlloc_bin = nlloc_bin
        self.nll_time_path = nlloc_times_path
        self.nlloc_template = nlloc_template
        self.nll_channel_hint = nll_channel_hint
        self.tmpdir = tmpdir
        self.double_pass = double_pass
        self.force_uncertainty = force_uncertainty
        self.P_uncertainty = P_uncertainty
        self.S_uncertainty = S_uncertainty
        self.P_time_residual_threshold = P_time_residual_threshold
        self.S_time_residual_threshold = S_time_residual_threshold
        self.nll_min_phase = nll_min_phase
        self.nll_verbose = nll_verbose

        # obs file to localize
        self.nll_obs_file = nll_obs_file

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

    def reloc_event(self, event):
        """
        Event relocalisation using a locator
        Returns a Catalog()
        """

        show_event(event, "****", header=True)
        orig = event.preferred_origin()
        channel_hint = io.StringIO()
        for arrival in orig.arrivals:
            pick = next(
                (p for p in event.picks if p.resource_id == arrival.pick_id), None
            )
            # keep channel info in channel_hint
            wfid = pick.waveform_id
            string = f"{wfid.network_code}_{wfid.station_code}_{wfid.location_code}_{wfid.channel_code}\n"
            channel_hint.write(string)
        channel_hint.seek(0)

        self.nll_obs_file = os.path.join(self.tmpdir, "nll_obs.txt")
        logger.debug(
            f"Writing nll_obs file to {self.nll_obs_file} in {self.tmpdir} directory."
        )
        event.write(self.nll_obs_file, format="NLLOC_OBS")
        self.nll_channel_hint = channel_hint
        cat = self.nll_localisation()
        return cat

    def nll_localisation(self, nll_obs_file=None, double_pass=None, pass_count=0):
        """
        Do the NLL stuff to localize event phases in nll_obs_file

        When double_pass is True, the localization is computed twice
        with picks/phases clean_up step.

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

        logger.debug(
            f"Localization of {nll_obs_file} using {self.nlloc_template} template."
        )
        nll_obs_file_basename = os.path.basename(nll_obs_file)

        tmp_path = tempfile.mkdtemp(dir=self.tmpdir)
        conf_file = os.path.join(tmp_path, f"{nll_obs_file_basename}.conf")

        # path + root filename
        output = os.path.join(tmp_path, nll_obs_file_basename)

        # Values to be substitued in the template
        tags = {
            "OBSFILE": nll_obs_file,
            "NLL_TIME_PATH": self.nll_time_path,
            "OUTPUT": output,
            "NLL_MIN_PHASE": self.nll_min_phase,
        }

        # Generate NLL configuration file
        try:
            self.replace(self.nlloc_template, conf_file, tags)
        except Exception as e:
            logger.error(e)
            return Catalog()

        # Localization
        cmde = f"{self.nlloc_bin} {conf_file}"
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

        # check from stdout if there is any missing station grid file
        # WARNING: cannot open grid buffer file: nll_times/auvergne-pyrocko2/auvergne.P.01x02.time.buf
        for line in result.stdout.splitlines():
            if "WARNING: cannot open grid buffer file" in line:
                logger.error(line)

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

        # there is always only one event in the catalog
        e = cat.events[0]
        o = e.preferred_origin()

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

        o.creation_info.agency_id = "RENASS"
        o.creation_info.author = "DBClust"
        o.evaluation_mode = "automatic"
        o.method_id = "NonLinLoc"
        o.earth_model_id = self.nlloc_template

        if self.force_uncertainty:
            for pick in e.picks:
                if "P" in pick.phase_hint or "p" in pick.phase_hint:
                    pick.time_errors.uncertainty = self.P_uncertainty
                elif "S" in pick.phase_hint or "s" in pick.phase_hint:
                    pick.time_errors.uncertainty = self.S_uncertainty

        # try a relocation
        if self.double_pass and pass_count == 0:
            logger.debug("Starting double pass relocation.")
            cat2 = cat.copy()
            event2 = cat2.events[0]
            event2 = self.cleanup_pick_phase(event2)
            new_nll_obs_file = nll_obs_file + ".2nd_pass"
            cat2.write(new_nll_obs_file, format="NLLOC_OBS")
            cat2 = self.nll_localisation(new_nll_obs_file, double_pass=self.double_pass, pass_count=1)

            if cat2:
                event2 = cat2.events[0]
                orig2 = event2.preferred_origin()
                # add this new origin to catalog and set it as preferred
                e.origins.append(orig2)
                e.preferred_origin_id = orig2.resource_id
                e.picks += event2.picks
            else:
                # can't relocate: set it to "not existing"
                e.event_type = "not existing"

        return cat

    def dask_get_localisations_from_nllobs_dir(self, OBS_PATH, append=True):
        """
        nll localisation and export to quakeml
        warning : network and channel are lost since they are not used by nll
                  use Phase() to get them.
        if append is True, the obtain catalog is appended to the NllLoc catalog

        returns a catalog
        """
        obs_files_pattern = os.path.join(OBS_PATH, "cluster-*.obs")
        logger.info(f"Localization of {obs_files_pattern}")

        #cluster = LocalCluster(silence_logs=logging.ERROR)
        # # n_workers=workers, hreads_per_worker=1, silence_logs=logging.ERROR, interface="lan",
        #with Client(cluster) as client: 
        b = db.from_sequence(glob.glob(obs_files_pattern), npartitions=multiprocessing.cpu_count())
        cat_results = b.map(
            lambda x: self.nll_localisation(x, double_pass=self.double_pass)
        ).compute()

        mycatalog = Catalog()
        for cat in cat_results:
            mycatalog += cat

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
        """
        nll localisation and export to quakeml
        warning : network and channel are lost since they are not used by nll
                  use Phase() to get them.
        if append is True, the obtain catalog is appended to the NllLoc catalog

        returns a catalog
        """
        obs_files_pattern = os.path.join(OBS_PATH, "cluster-*.obs")
        logger.info(f"Localization of {obs_files_pattern}")

        mycatalog = Catalog()
        for nll_obs_file in sorted(glob.glob(obs_files_pattern)):
            # localization
            cat = self.nll_localisation(nll_obs_file, double_pass=self.double_pass)
            if not cat:
                logger.debug(f"No loc obtained for {nll_obs_file}:/")
                continue
            mycatalog += cat

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

            if isclose(arrival.time_weight, 0, abs_tol=0.001) or bad_time_residual:
                pick = next(
                    (p for p in event.picks if p.resource_id == arrival.pick_id), None
                )
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
                logger.info(f"Duplicated pick detected [{p.waveform_id.get_seed_string()}, {a.phase}, {p.time}]... removing the one with highest residual")
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
        orig.quality.used_station_count = NllLoc.get_used_station_count(event)
        return event

    @staticmethod
    def get_used_station_count(event):
        station_list = []
        origin = event.preferred_origin()
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
            "Use the command below to get it from the seed files you got the picks from :\n"
            "> msi -T ${MSEED_DIR}/*  | tail -n +2 | head -n-1 | cut -f1 -d' ' | sort -u > chan.txt\n"
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
                        logger.warning(f"Duplicated network code for station {sta}")
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
                        logger.warning(f"Duplicated location code for {net}.{sta}")
                        logger.warning(f"    using the first one {loc.iloc[0]}")
                    loc = loc.iloc[0]
                    p.waveform_id.location_code = loc

                chan = df[(df["sta"] == sta) & (df["net"] == net) & (df["loc"] == loc)][
                    "chan"
                ].drop_duplicates()
                if len(chan) == 0:
                    logger.warning("Channel code not found for {net}.{sta}")
                elif len(chan) != 1:
                    logger.warning(f"Duplicated channel code for {net}.{sta}.{loc}")
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
        print("Text, T0, lat, lon, depth, RMS, sta_count, phase_count, gap1, gap2")
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
                    f"{o.quality.azimuthal_gap:.1f}",
                    f"{o.quality.secondary_azimuthal_gap:.1f}"
                    if o.quality.secondary_azimuthal_gap
                    else "-",
                ],
            )
        )
    )


def _simple_test():
    nll_obs_file = "../test/cluster-0.obs"
    nlloc_times_path = "/Users/marc/Dockers/routine/nll/data/times"
    nlloc_template = "../nll_template/nll_haslach_template.conf"
    nlloc_bin = "NLLoc"
    nll_channel_hint = "../test/chan.txt"

    loc = NllLoc(
        nlloc_bin,
        nlloc_times_path,
        nlloc_template,
        nll_channel_hint=nll_channel_hint,
        nll_obs_file=nll_obs_file,
        tmpdir="/tmp",
        nll_verbose=False,
    )
    print(loc.catalog)
    loc.show_localizations()


def _multiple_test():
    nlloc_bin = "NLLoc"
    nlloc_times_path = "/Users/marc/Dockers/routine/nll/data/times"
    nlloc_template = "../nll_template/nll_haslach_template.conf"

    obs_path = "../test"
    qml_path = "../test"
    nll_channel_hint = "../test/chan.txt"

    loc = NllLoc(
        nlloc_bin,
        nlloc_times_path,
        nlloc_template,
        nll_channel_hint=nll_channel_hint,
        tmpdir="/tmp",
        nll_verbose=False,
    )
    cat = loc.get_localisations_from_nllobs_dir(obs_path)
    print(cat)
    loc.show_localizations()
    # logger.info("Writing all.xml")
    # catalog.write("all.xml", format="QUAKEML")


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)

    # logger.info("")
    # logger.info("++++++++++++++++ Reloc test (catalog)")
    # _cat_reloc("../DataChambon/qml/chambon.qml", force_uncertainty=True)
    # _cat_reloc("../test/chambon.qml")
    # sys.exit()

    logger.info("")
    logger.info("++++++++++++++++ Reloc test (fdsnws-event)")
    event_id = "fr2023lahzgh"
    _event_reloc_test(
        event_id, force_uncertainty=True, P_uncertainty=0.1, S_uncertainty=0.2
    )

    logger.info("++++++++++++++++ Simple test")
    _simple_test()

    logger.info("")
    logger.info("++++++++++++++++ Multiple test")
    _multiple_test()
