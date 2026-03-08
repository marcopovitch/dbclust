#!/usr/bin/env python
"""Locate seismic events from CSV picks using NonLinLoc."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd
from obspy import UTCDateTime
from obspy.core.event import Catalog, Comment, Event, Pick, ResourceIdentifier
from obspy.core.event import CreationInfo
from obspy.core.event.base import WaveformStreamID

try:
    from dbclust.config import DBClustConfig
except ImportError:  # pragma: no cover - optional dependency at runtime only
    DBClustConfig = None  # type: ignore

from dbclust.localization import LocalizationError, NllLoc

LOGGER = logging.getLogger("dbclust.locate")
REQUIRED_COLUMNS = {
    "station_id",
    "channel",
    "phase_type",
    "phase_time",
    "phase_score",
    "phase_evaluation",
    "phase_method",
    "event_id",
    "agency",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locate an event from CSV picks using NonLinLoc"
    )
    parser.add_argument("picks_csv", help="CSV file containing picks")
    parser.add_argument("nll_template", help="NonLinLoc template file to use")
    parser.add_argument("output", help="Output QuakeML file path")
    parser.add_argument(
        "-c",
        "--conf",
        dest="config",
        default=None,
        help="Path to dbclust YAML configuration (optional)",
    )
    parser.add_argument(
        "--nll-bin",
        dest="nll_bin",
        default=None,
        help="Path to the NonLinLoc binary",
    )
    parser.add_argument(
        "--scat2latlon-bin",
        dest="scat2latlon_bin",
        default=None,
        help="Path to scat2latlon binary (optional)",
    )
    parser.add_argument(
        "--nll-time-path",
        dest="nll_time_path",
        default=None,
        help="Path to NonLinLoc time grids",
    )
    parser.add_argument(
        "--loc-method",
        dest="loc_method",
        default=None,
        help="NonLinLoc location method (default from config or EDT_OT_WT_ML)",
    )
    parser.add_argument(
        "--min-phase",
        dest="min_phase",
        type=int,
        default=None,
        help="Minimum number of phases required (default from config or 4)",
    )
    parser.add_argument(
        "--double-pass",
        dest="double_pass",
        action="store_true",
        help="Force double-pass localization",
    )
    parser.add_argument(
        "--single-pass",
        dest="single_pass",
        action="store_true",
        help="Force single-pass localization",
    )
    parser.add_argument(
        "--tmp-dir",
        dest="tmp_dir",
        default=None,
        help="Directory for temporary NonLinLoc files",
    )
    parser.add_argument(
        "--p-uncertainty",
        dest="p_uncertainty",
        type=float,
        default=0.1,
        help="Uncertainty (s) assigned to P picks",
    )
    parser.add_argument(
        "--s-uncertainty",
        dest="s_uncertainty",
        type=float,
        default=0.2,
        help="Uncertainty (s) assigned to S picks",
    )
    parser.add_argument(
        "--loglevel",
        dest="loglevel",
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--agency-id",
        dest="agency_id",
        default="DBCLUST",
        help="Agency identifier used in metadata when no config is provided",
    )
    parser.add_argument(
        "--author",
        dest="author",
        default="locate-script",
        help="Author written in metadata when no config is provided",
    )
    parser.add_argument(
        "--p-time-residual-threshold",
        dest="p_time_residual_threshold",
        type=float,
        default=None,
        help="P phase time residual threshold (s) for pick cleanup (default: None)",
    )
    parser.add_argument(
        "--s-time-residual-threshold",
        dest="s_time_residual_threshold",
        type=float,
        default=None,
        help="S phase time residual threshold (s) for pick cleanup (default: None)",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(stream=sys.stdout, level=numeric_level, force=True)
    LOGGER.setLevel(numeric_level)


def read_picks_dataframe(picks_file: str) -> pd.DataFrame:
    if not os.path.exists(picks_file):
        raise FileNotFoundError(f"Picks file {picks_file} does not exist")

    df = pd.read_csv(picks_file)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV file is missing required columns: {', '.join(sorted(missing))}"
        )

    df["phase_time"] = pd.to_datetime(df["phase_time"], utc=True)
    return df


def dataframe_to_event(
    df: pd.DataFrame,
    p_uncertainty: float,
    s_uncertainty: float,
) -> Event:
    event = Event(resource_id=ResourceIdentifier(f"smi:local/{uuid.uuid4()}"))

    for row in df.itertuples(index=False):
        station_id = str(row.station_id)
        station_parts = station_id.split(".")
        network = station_parts[0] if len(station_parts) > 0 else ""
        station = station_parts[1] if len(station_parts) > 1 else ""
        location = station_parts[2] if len(station_parts) > 2 else ""

        channel = str(getattr(row, "channel", "")).strip()
        if not channel and len(station_parts) > 3:
            channel = station_parts[3]

        # Handle channel containing "location.channel" (e.g. "00.HHZ")
        if "." in channel:
            chan_parts = channel.split(".", 1)
            if not location:
                location = chan_parts[0]
            channel = chan_parts[1]

        waveform_id = WaveformStreamID(
            network_code=network,
            station_code=station,
            location_code=location,
            channel_code=channel[:3] if channel else "",
        )

        pick = Pick(
            time=UTCDateTime(row.phase_time.to_pydatetime()),
            waveform_id=waveform_id,
            phase_hint=str(row.phase_type).strip(),
        )

        evaluation = str(row.phase_evaluation).lower() if row.phase_evaluation else ""
        if evaluation in {"manual", "automatic"}:
            pick.evaluation_mode = evaluation

        method_id = str(row.phase_method).strip() if row.phase_method else ""
        if method_id:
            try:
                pick.method_id = ResourceIdentifier(method_id)
            except Exception:  # pragma: no cover - depends on user input
                LOGGER.warning("Invalid method identifier '%s', skipping", method_id)

        if pick.phase_hint.upper().startswith("P"):
            pick.time_errors.uncertainty = p_uncertainty
        elif pick.phase_hint.upper().startswith("S"):
            pick.time_errors.uncertainty = s_uncertainty

        pick.creation_info = CreationInfo(agency_id=getattr(row, "agency", None) or None)

        # Store phase_score as a probability comment (same format as clusterize.py)
        phase_score = getattr(row, "phase_score", None)
        if phase_score is not None:
            agency = getattr(row, "agency", None) or "undefined"
            pick.comments.append(
                Comment(
                    text='{"probability": {"name": "%s", "value": %.2f}}'
                    % (agency, float(phase_score))
                )
            )

        event.picks.append(pick)

    if not event.picks:
        raise ValueError("No picks could be created from CSV input")

    return event


def resolve_value(
    cli_value: Optional[str], cfg_value: Optional[str], env_var: Optional[str]
) -> Optional[str]:
    if cli_value:
        return cli_value
    if cfg_value:
        return cfg_value
    if env_var:
        return env_var
    return None


def build_locator(
    args: argparse.Namespace,
    cfg: Optional[DBClustConfig],
) -> NllLoc:
    cfg_nll = getattr(cfg, "nll", None)
    cfg_reloc = getattr(cfg, "relocation", None)
    cfg_cluster = getattr(cfg, "cluster", None)
    cfg_catalog = getattr(cfg, "catalog", None)

    nll_bin = resolve_value(args.nll_bin, getattr(cfg_nll, "nlloc_bin", None), os.getenv("NLL_BIN"))
    nll_time_path = resolve_value(
        args.nll_time_path, getattr(cfg_nll, "time_path", None), os.getenv("NLL_TIME_PATH")
    )
    scat2latlon_bin = resolve_value(
        args.scat2latlon_bin,
        getattr(cfg_nll, "scat2latlon_bin", None),
        os.getenv("SCAT2LATLON_BIN"),
    )

    if not nll_bin:
        raise ValueError("NonLinLoc binary path must be provided via --nll-bin, config, or NLL_BIN")
    if not nll_time_path:
        raise ValueError(
            "NonLinLoc time path must be provided via --nll-time-path, config, or NLL_TIME_PATH"
        )
    if not scat2latlon_bin:
        scat2latlon_bin = nll_bin  # only used if scatter output is requested

    loc_method = args.loc_method or getattr(cfg_nll, "loc_method", "EDT_OT_WT_ML")
    min_phase = args.min_phase or getattr(cfg_nll, "min_phase", 4)

    if args.single_pass:
        double_pass = False
    elif args.double_pass:
        double_pass = True
    else:
        double_pass = getattr(cfg_reloc, "double_pass", False)

    tmp_dir = args.tmp_dir or getattr(getattr(cfg, "file", None), "tmp_path", None)
    tmp_dir = tmp_dir or tempfile.gettempdir()

    if cfg:
        quakeml_settings = asdict(cfg.quakeml)
    else:
        quakeml_settings = {
            "agency_id": args.agency_id,
            "author": args.author,
            "evaluation_mode": "automatic",
            "method_id": "NonLinLoc",
            "model_id": None,
        }

    locator = NllLoc(
        nll_bin=nll_bin,
        scat2latlon_bin=scat2latlon_bin,
        nll_times_path=nll_time_path,
        nll_template=args.nll_template,
        nll_default_template=args.nll_template,
        nll_min_phase=min_phase,
        loc_method=loc_method,
        tmpdir=tmp_dir,
        double_pass=double_pass,
        gap_dist_max_km=getattr(cfg_reloc, "gap_dist_max_km", None),
        closest_station_dist_km=getattr(cfg_reloc, "closest_station_dist_km", None),
        dist_km_cutoff=getattr(cfg_reloc, "dist_km_cutoff", None),
        use_deactivated_arrivals=getattr(cfg_reloc, "use_deactivated_arrivals", False),
        keep_manual_picks=getattr(cfg_reloc, "keep_manual_picks", False),
        min_station_with_P_and_S=getattr(cfg_cluster, "min_station_with_P_and_S", 0),
        min_station_score=getattr(cfg_cluster, "min_station_score", None),
        quakeml_settings=quakeml_settings,
        keep_scat=getattr(cfg_nll, "enable_scatter", False),
        zones=getattr(cfg, "zones", None),
        min_score_threshold_pick_zone=getattr(cfg_reloc, "min_score_threshold_pick_zone", 0.5),
        use_pick_zone=getattr(cfg_reloc, "use_pick_zone", False),
        enable_cleanup_pick_zone=getattr(cfg_reloc, "enable_cleanup_pick_zone", False),
        enable_relabel_pick_zone=getattr(cfg_reloc, "enable_relabel_pick_zone", False),
        min_dist_relabel_deg=getattr(cfg_reloc, "min_dist_relabel_deg", 0.0),
        min_time_weight=getattr(cfg_reloc, "min_time_weight", None),
        keep_not_existing_event=getattr(cfg_catalog, "keep_not_existing_event", False),
        P_time_residual_threshold=args.p_time_residual_threshold or getattr(cfg_reloc, "P_time_residual_threshold", None),
        S_time_residual_threshold=args.s_time_residual_threshold or getattr(cfg_reloc, "S_time_residual_threshold", None),
    )

    return locator


def ensure_file_exists(path: str, description: str) -> str:
    if not path:
        raise ValueError(f"Missing {description}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{description} '{path}' does not exist")
    return path


def run_localization(
    locator: NllLoc,
    event: Event,
    output_quakeml: str,
) -> None:
    with tempfile.TemporaryDirectory(dir=locator.tmpdir) as tmp_dir:
        obs_file = Path(tmp_dir) / "picks.obs"
        catalog = Catalog(events=[event])
        catalog.write(str(obs_file), format="NLLOC_OBS")

        LOGGER.info("Running NonLinLoc (template: %s)", locator.nll_template)
        cat = locator.nll_localisation(nll_obs_file=str(obs_file), picks=event.picks)

        if not cat or len(cat.events) == 0:
            raise LocalizationError("NonLinLoc did not return any event")

        output_path = Path(output_quakeml)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cat.write(str(output_path), format="QUAKEML")
        LOGGER.info("Wrote QuakeML file to %s", output_path)


def load_config(path: Optional[str]) -> Optional[DBClustConfig]:
    if not path:
        return None
    if DBClustConfig is None:
        raise ImportError("dbclust.config.DBClustConfig is required but not available")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file {path} does not exist")
    return DBClustConfig(path, config_type="reloc")


def main() -> None:
    try:
        args = parse_arguments()
        setup_logging(args.loglevel)
        cfg = load_config(args.config)

        args.nll_template = ensure_file_exists(args.nll_template, "NonLinLoc template")
        df = read_picks_dataframe(args.picks_csv)
        event = dataframe_to_event(df, args.p_uncertainty, args.s_uncertainty)

        locator = build_locator(args, cfg)
        run_localization(locator, event, args.output)
    except Exception as exc:  # pragma: no cover - CLI error handling
        LOGGER.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
