#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression tests for BUGS_COHERENCE_REVIEW_2026-10-09.md, batch 1.

Covers:
- C1 (false positive, kept as a guard): Zones.load_zones picks the velocity
  profile matching the zone, not the last one of the list
- C2/Co6: the runner reads the JSON-lines checkpoint written by the executors
  and only merges temp DBs of completed jobs
- C3: a crashed task (result None) is not recorded as completed, while a job
  that ran to completion with dropped picks (result False) is
- stale/invalid checkpoints are reset instead of being appended to
"""
import json
import logging
from types import SimpleNamespace

import pytest

from dbclust import runner
from dbclust.config import Zone
from dbclust.config import Zones
from dbclust.executors.base import ExecutorBase
from dbclust.executors.base import config_fingerprint
from dbclust.executors.base import read_completed_checkpoint

logging.disable(logging.WARNING)


def make_cfg(tmp_path, n_partitions=4, start="2020-01-01"):
    return SimpleNamespace(
        pick=SimpleNamespace(start=start, end="2020-01-02"),
        time=SimpleNamespace(time_window=10, overlap_window=60),
        parallel=SimpleNamespace(
            task_profiles_path=str(tmp_path / "task_profiles.csv"),
            execution_summary_path=str(tmp_path / "execution_summary.csv"),
            task_profiles_reference_path=None,
            n_workers=n_partitions,
            oversubscription_factor=1,
            partition_duration="6h",
            time_partitions=[(f"s{i}", f"e{i}") for i in range(n_partitions)],
        ),
        catalog=SimpleNamespace(
            qml_path=str(tmp_path),
            enable_sqlite=True,
            temp_db_dir=str(tmp_path / "tmpdb"),
            sqlite_db_path=str(tmp_path),
            sqlite_db_fullpath=str(tmp_path / "final.db"),
            keep_temp_db_after_merge=True,
        ),
    )


class FakeExecutor(ExecutorBase):
    """Executor whose tasks return a predefined result per job index."""

    def __init__(self, cfg, outcomes):
        super().__init__(cfg)
        self.outcomes = outcomes

    @property
    def name(self):
        return "fake"

    def initialize(self):
        pass

    def submit_task(self, job_index):
        return job_index

    def wait_for_results(self, futures):
        for idx in futures:
            yield (idx, self.outcomes[idx], 0.0, 0.0)

    def cleanup(self):
        pass


def checkpoint_path(tmp_path):
    return str(tmp_path / "task_profiles.completed.json")


# --- C1 -------------------------------------------------------------------


def test_load_zones_uses_matching_velocity_profile():
    profiles = [
        SimpleNamespace(name="alps", template_file="alps.template"),
        SimpleNamespace(name="rhine", template_file="rhine.template"),
    ]
    square = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]
    zones = Zones(zones=[Zone(name="z1", velocity_profile="alps", polygon=square)])

    zones.load_zones(SimpleNamespace(velocity_profiles=profiles))

    row = zones.polygons.iloc[0]
    assert row["velocity_profile"] == "alps"
    assert row["template"] == "alps.template"


def test_load_zones_unknown_profile_raises():
    zones = Zones(zones=[Zone(name="z1", velocity_profile="nope", polygon=[[0, 0], [0, 1], [1, 1]])])
    with pytest.raises(ValueError, match="nope"):
        zones.load_zones(SimpleNamespace(velocity_profiles=[]))


# --- checkpoint format -------------------------------------------------------


def test_read_checkpoint_formats(tmp_path):
    path = checkpoint_path(tmp_path)
    assert read_completed_checkpoint(path, "fp") == (None, "missing")

    with open(path, "w") as f:
        f.write('{"fingerprint": "fp"}\n{"done": 3}\n{"done": 1}\ngarbage\n')
    assert read_completed_checkpoint(path, "fp") == ({1, 3}, "ok")
    assert read_completed_checkpoint(path, "other") == (None, "stale")

    # former single-object format is no longer supported
    with open(path, "w") as f:
        json.dump({"fingerprint": "fp", "done": [0, 2]}, f)
    assert read_completed_checkpoint(path, "fp") == (None, "invalid")

    with open(path, "w") as f:
        json.dump([0, 1], f)
    assert read_completed_checkpoint(path, "fp") == (None, "invalid")


def test_stale_checkpoint_is_reset_not_appended(tmp_path):
    cfg = make_cfg(tmp_path)
    path = checkpoint_path(tmp_path)
    with open(path, "w") as f:
        f.write('{"fingerprint": "old"}\n{"done": 0}\n')

    executor = FakeExecutor(cfg, {})
    assert executor._load_completed() == set()
    executor._mark_completed(2)

    assert read_completed_checkpoint(path, config_fingerprint(cfg)) == ({2}, "ok")


# --- C3 + C2 end-to-end --------------------------------------------------------


def test_crashed_task_not_marked_and_not_merged(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    # 0: ok, 1: crashed, 2: completed with dropped picks, 3: ok
    outcomes = {0: True, 1: None, 2: False, 3: True}
    results = FakeExecutor(cfg, outcomes).run()
    assert sorted(results, key=str) == sorted(outcomes.values(), key=str)

    completed = runner._load_completed_indices(cfg)
    assert completed == {0, 2, 3}

    temp_dir = tmp_path / "tmpdb"
    temp_dir.mkdir()
    for i in range(4):
        (temp_dir / f"tmp_worker_{i}.db").touch()

    merged = []
    monkeypatch.setattr(
        runner, "merge_databases", lambda paths, *a, **kw: merged.extend(paths)
    )
    assert runner.finalize_sqlite(cfg, completed_job_indices=completed)
    assert sorted(p.rsplit("/", 1)[1] for p in merged) == [
        "tmp_worker_0.db",
        "tmp_worker_2.db",
        "tmp_worker_3.db",
    ]


def test_resume_reruns_only_crashed_task(tmp_path):
    cfg = make_cfg(tmp_path)
    FakeExecutor(cfg, {0: True, 1: None, 2: True, 3: True}).run()

    executor = FakeExecutor(cfg, {1: True})
    assert executor.run() == [True]
    assert runner._load_completed_indices(cfg) == {0, 1, 2, 3}


def test_runner_without_checkpoint_merges_nothing(tmp_path):
    cfg = make_cfg(tmp_path)
    assert runner._load_completed_indices(cfg) == set()
