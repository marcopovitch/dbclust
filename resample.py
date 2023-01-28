#!/usr/bin/env python
import sys
import os
import argparse
import glob
import logging
from obspy import read

# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("resample")
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
        default=None,
        dest="input_dir",
        help="input directory",
        type=str,
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        default=None,
        dest="output_dir",
        help="output directory",
        type=str,
    )

    parser.add_argument(
        "-s",
        "--sr",
        default=100,
        dest="sample_rate",
        help="sample rate",
        type=float,
    )

    args = parser.parse_args()
    if not args.input_dir or not args.output_dir:
        parser.print_help()
        sys.exit(255)

    os.makedirs(args.output_dir)

    for f_in in glob.glob(f"{args.input_dir}/*"):
        try:
            st = read(f_in)
        except Exception as e:
            logger.error(f"Read failed for {f_in} ({e})")
            continue
        st1 = st.select(channel="HH?")
        st2 = st.select(channel="EH?")
        st3 = st.select(channel="HN?")
        st = st1 + st2 + st3

        if not st:
            continue

        # subsample channel to sample_rate Hz
        for tr in st:
            if tr.stats.sampling_rate > args.sample_rate:
                logger.debug(
                    f"Resampling {tr.id} from {tr.stats.sampling_rate} to {args.sample_rate} Hz"
                )
                try:
                    tr.resample(sampling_rate=args.sample_rate)
                except Exception as e:
                    logger.error(f"Resample failed for {tr.id} ({e})")
            else:
                logger.debug(
                    f"{tr.id}, sample rate ({tr.stats.sampling_rate}) <= {args.sample_rate} Hz: nothing to do !"
                )

        f_out = os.path.join(args.output_dir, os.path.basename(f_in))
        st.write(f"{f_out}", format="MSEED")
