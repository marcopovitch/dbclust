#!/usr/bin/env python3
"""
Test script to validate that samples/config.yml and samples/config_slurm.yml
still load through DBClustConfig.

Regression guard for T2 (BUGS_COHERENCE_REVIEW_2026-09-17.md): the sample
configs previously drifted out of sync with DBClustConfig and failed to load
at all (dacite.exceptions.MissingValueError). This script builds a throwaway
directory tree matching the paths referenced by the samples (data_path,
picks file, NLL binaries/templates) so DBClustConfig.__post_init__'s
filesystem checks pass, then loads each sample and asserts success.
"""
import os
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

from dbclust.config import DBClustConfig

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")

CSV_HEADER = (
    "station_id,channel,phase_type,phase_time,phase_score,"
    "phase_evaluation,phase_method,event_id,agency\n"
)
CSV_ROW = "FR.TEST,HHZ,P,2022-06-01T00:00:00.000Z,0.9,automatic,phasenet,,TEST\n"

TEMPLATE_NAMES = [
    "nll_alpes_template.conf",
    "nll_auvergne_template.conf",
    "nll_haslach-0.2_template.conf",
    "nll_pyrenees_template.conf",
    "nll_rittershoffen_template.conf",
    "nll_vendenheim_template.conf",
]


def build_fake_tree(root: str) -> None:
    os.makedirs(os.path.join(root, "picks"), exist_ok=True)
    os.makedirs(os.path.join(root, "qml"), exist_ok=True)
    os.makedirs(os.path.join(root, "db"), exist_ok=True)
    os.makedirs(os.path.join(root, "vmodels"), exist_ok=True)

    with open(os.path.join(root, "picks", "picks.csv"), "w") as f:
        f.write(CSV_HEADER)
        f.write(CSV_ROW)

    for name in TEMPLATE_NAMES:
        open(os.path.join(root, name), "w").close()


def rewrite_sample(sample_path: str, root: str, dest_path: str) -> None:
    with open(sample_path) as f:
        text = f.read()

    true_bin = shutil.which("true") or "/usr/bin/true"
    text = text.replace("/datadir", root)
    text = text.replace("/opt/nll/bin/NLLoc", true_bin)
    text = text.replace("/opt/nll/bin/scat2latlon", true_bin)
    text = re.sub(r"/nll/times", root, text)
    text = re.sub(r"/nll/nll_template", root, text)

    with open(dest_path, "w") as f:
        f.write(text)


def check_sample(name: str) -> bool:
    sample_path = os.path.join(SAMPLES_DIR, name)
    with tempfile.TemporaryDirectory(prefix="dbclust_sample_test_") as tmp:
        build_fake_tree(tmp)
        rewritten = os.path.join(tmp, name)
        rewrite_sample(sample_path, tmp, rewritten)
        try:
            DBClustConfig(rewritten)
        except Exception as e:
            print(f"✗ {name}: failed to load: {type(e).__name__}: {e}")
            return False
    print(f"✓ {name}: loaded OK")
    return True


def main() -> bool:
    ok = True
    for name in ["config.yml", "config_slurm.yml"]:
        ok = check_sample(name) and ok
    return ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
