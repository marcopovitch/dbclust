#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression test for BUGS_COHERENCE_REVIEW_2026-10-09.md, D9.

deduplicate_cross_partition_events() deletes the losing event with
DELETE FROM events and relies on ON DELETE CASCADE for its origins and picks.
SQLite ignores the cascade unless foreign keys are enabled on the connection,
which is off by default (e.g. Utils/dedup_cross_partition.py).
"""
import logging
import sqlite3

from dbclust.parallel_import import deduplicate_cross_partition_events

logging.disable(logging.WARNING)


def make_db():
    conn = sqlite3.connect(":memory:")  # foreign keys OFF, as in sqlite3.connect()
    conn.executescript(
        """
        CREATE TABLE events (event_id TEXT PRIMARY KEY);
        CREATE TABLE origins (
            id TEXT PRIMARY KEY,
            event_id TEXT REFERENCES events(event_id) ON DELETE CASCADE,
            time TEXT,
            used_phase_count INTEGER
        );
        CREATE TABLE picks (
            id TEXT PRIMARY KEY,
            event_id TEXT REFERENCES events(event_id) ON DELETE CASCADE,
            source_event_id TEXT,
            evaluation_mode TEXT,
            station_name TEXT,
            phase_hint TEXT,
            pick_time TEXT
        );
        """
    )
    # Same physical event localised by two jobs: manual picks share 2 catalog ids
    for ev, offset in (("ev_a", 0), ("ev_b", 1)):
        conn.execute("INSERT INTO events VALUES (?)", (ev,))
        conn.execute(
            "INSERT INTO origins VALUES (?, ?, '2020-01-01T00:00:00', 10)",
            (f"o_{ev}", ev),
        )
        for i, src in enumerate(("cat1", "cat2")):
            conn.execute(
                "INSERT INTO picks VALUES (?, ?, ?, 'manual', ?, 'P', ?)",
                (f"p_{ev}_{i}", ev, src, f"STA{i}", f"2020-01-01T00:00:0{i + offset}"),
            )
    conn.commit()
    return conn


def test_dedup_deletes_children_of_removed_event():
    conn = make_db()

    result = deduplicate_cross_partition_events(conn)

    assert result["n_removed"] == 1
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    orphans = conn.execute(
        """
        SELECT (SELECT COUNT(*) FROM picks WHERE event_id NOT IN (SELECT event_id FROM events))
             + (SELECT COUNT(*) FROM origins WHERE event_id NOT IN (SELECT event_id FROM events))
        """
    ).fetchone()[0]
    assert orphans == 0
