#!/usr/bin/env python
import sys
import os
import logging
import glob
import pathlib
import subprocess
import tempfile
import pandas as pd

# from tqdm import tqdm
from obspy import Catalog
from obspy import read_events
from jinja2 import Template

# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("localization")
logger.setLevel(logging.INFO)


class NllLoc(object):
    def __init__(
        self,
        nllocbin,
        nlloc_template,
        nll_obs_file=None,
        nll_channel_hint=None,
        tmpdir="/tmp",
    ):

        # define locator
        self.nllocbin = nllocbin
        self.nlloc_template = nlloc_template
        self.nll_channel_hint = nll_channel_hint
        self.tmpdir = tmpdir

        # obs file to localize
        self.nll_obs_file = nll_obs_file

        # keep track of cluster affiliation 
        self.event_cluster_mapping = {}

        # localization
        if self.nll_obs_file:
            self.catalog = self.nll_localisation(nll_obs_file)
        else:
            self.catalog = Catalog()

        self.nb_events = len(self.catalog)


    def nll_localisation(self, nll_obs_file):

        nll_obs_file_basename = os.path.basename(nll_obs_file)

        # with tempfile.TemporaryDirectory(dir=tmpdir) as tmp_path:
        tmp_path = tempfile.mkdtemp(dir=self.tmpdir)
        conf_file = os.path.join(tmp_path, f"{nll_obs_file_basename}.conf")

        # path + root filename
        output = os.path.join(tmp_path, nll_obs_file_basename)

        # Values to be substitued in the template
        tags = {
            "OBSFILE": nll_obs_file,
            "OUTPUT": output,
        }

        # Generate NLL configuration file
        try:
            self.replace(self.nlloc_template, conf_file, tags)
        except Exception as e:
            logger.error(e)
            return None

        # Localization
        cmde = f"{self.nllocbin} {conf_file}"
        logger.debug(cmde)
        try:
            res = subprocess.call(
                cmde, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
            )
        except Exception as e:
            logger.error(e)
            return None
        else:
            logger.debug(f"res = {res}")

        # Read results
        nll_output = os.path.join(tmp_path, "last.hyp")
        try:
            cat = read_events(nll_output)
        except Exception as e:
            logger.debug(e)
            cat = None
        else:
            # check for nan value in uncertainty
            e = cat.events[0]
            o = e.preferred_origin()
            if "nan" in [
                str(o.latitude_errors.uncertainty),
                str(o.longitude_errors.uncertainty),
                str(o.depth_errors.uncertainty),
            ]:
                cat = None

        return cat

    def get_localisations_from_nllobs_dir(
        self,
        OBS_PATH,
        QML_PATH,
    ):
        """
        nll localisation and export to quakeml
        warning : network and channel are lost since they are not used by nll
                  use Phase() to get them.
        """
        obs_files_pattern = os.path.join(OBS_PATH, "cluster-*.obs")
        logger.info(f"Localization of {obs_files_pattern} to {QML_PATH}")

        for nll_obs_file in sorted(glob.glob(obs_files_pattern)):
            logger.info(
                f"Localization of {nll_obs_file} using {self.nlloc_template} nlloc template."
            )
            nll_obs_file_basename = os.path.basename(nll_obs_file)
            os.makedirs(QML_PATH, exist_ok=True)
            qmlfile = os.path.join(
                QML_PATH, pathlib.PurePath(nll_obs_file_basename).stem
            )

            # localization
            cat = self.nll_localisation(nll_obs_file)

            if cat:
                if self.nll_channel_hint:
                    logger.debug(self.nll_channel_hint)
                    cat = self.fix_wfid(cat, self.nll_channel_hint)
                else:
                    logger.warning("No nll_channel_hint file provided !")

                # override default values
                for e in cat.events:
                    e.creation_info.author = ""
                    o = e.preferred_origin()
                    o.creation_info.agency_id = "RENASS"
                    o.creation_info.author = "DBClust"
                    o.evaluation_mode = "automatic"
                    o.method_id = "NonLinLoc"
                    o.earth_model_id = self.nlloc_template
                    # count the stations used with weight > 0
                    o.quality.used_station_count = len(
                        [a.time_weight for a in o.arrivals if a.time_weight]
                    )
                    # o.quality.phase_station_count =

                logger.debug(f"Writing {qmlfile}.xml")
                cat.write(f"{qmlfile}.xml", format="QUAKEML")
                cat.write(f"{qmlfile}.sc3ml", format="SC3ML")
                self.event_cluster_mapping[e.resource_id.id] = nll_obs_file
            else:
                logger.debug(f"No loc obtained for {qmlfile}:/")
                continue
            self.catalog += cat

        # sort events by time
        self.nb_events = len(self.catalog)
        if self.nb_events > 1:
            self.catalog.events = sorted(
                self.catalog.events, key=lambda e: e.preferred_origin().time
            )
        return self

    @staticmethod
    def replace(templatefile, outfilename, tags):
        with open(templatefile) as file_:
            template = Template(file_.read())
        t = template.render(tags)
        with open(outfilename, "w") as out_fh:
            out_fh.write(t)
            logger.debug(
                f"NLLoc template file {templatefile} rendered to {outfilename}"
            )

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
        print("OBSfile, T0, lat, lon, depth, RMS, sta_count, phase_count, gap1, gap2")
        for e in self.catalog.events:
            try:
                nll_obs = self.event_cluster_mapping[e.resource_id.id]
            except:
                nll_obs = None

            o = e.preferred_origin()
            print(
                ", ".join(
                    map(
                        str,
                        [
                            nll_obs,
                            o.time,
                            f"{o.latitude:.3f}",
                            f"{o.longitude:.3f}",
                            f"{o.depth:.1f}",
                            o.quality.standard_error,
                            o.quality.used_station_count,
                            o.quality.used_phase_count,
                            f"{o.quality.azimuthal_gap:.1f}",
                            f"{o.quality.secondary_azimuthal_gap:.1f}",
                        ],
                    )
                )
            )


def _simple_test():
    nll_obs_file = "../test/cluster-0.obs"
    nlloc_template = "../nll_template/nll_haslach_template.conf"
    nllocbin="NLLoc" 

    loc = NllLoc(nllocbin, nlloc_template, nll_obs_file=nll_obs_file, tmpdir="/tmp")
    print(loc.catalog)
    loc.show_localizations()


def _multiple_test():
    obs_path = "../test"
    qml_path = "../test"
    nlloc_template = "../nll_template/nll_haslach_template.conf"
    nll_channel_hint = "../test/chan.txt"
    nllocbin="NLLoc" 

    locator = NllLoc(nllocbin, nlloc_template, nll_channel_hint=nll_channel_hint, tmpdir="/tmp") 
    loc = locator.get_localisations_from_nllobs_dir( obs_path, qml_path)
    print(loc.catalog)
    loc.show_localizations()
    # logger.info("Writing all.xml")
    # catalog.write("all.xml", format="QUAKEML")


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logger.info("++++++++++++++++Simple test")
    _simple_test()
    logger.info("")
    logger.info("++++++++++++++++Multiple test")
    _multiple_test()
