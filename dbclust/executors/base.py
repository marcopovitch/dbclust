"""Abstract base class for all DBClust executors.

This module defines the ExecutorBase class that all parallel execution
backends must inherit from.
"""

import csv
import json
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from dbclust.config import DBClustConfig
from dbclust.core import CSV_FIELDNAMES

logger = logging.getLogger("dbclust")


class ExecutorBase(ABC):
    """Abstract base class for parallel execution backends.

    All executor implementations (Ray, Parsl, Dask) must inherit from this
    class and implement the abstract methods.

    Attributes:
        cfg: DBClust configuration object.
        profile_csv_path: Path to the CSV file for task profiling.
    """

    def __init__(self, cfg: DBClustConfig):
        """Initialize the executor.

        Args:
            cfg: DBClust configuration containing parallel execution settings.
        """
        self.cfg = cfg
        self.profile_csv_path = cfg.parallel.task_profiles_path or os.path.join(
            cfg.catalog.qml_path, "task_profiles.csv"
        )
        self.summary_csv_path = cfg.parallel.execution_summary_path or os.path.join(
            cfg.catalog.qml_path, "execution_summary.csv"
        )
        self.run_start_time = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this executor."""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the executor (Ray init, Parsl load, etc.).

        This method should set up the parallel execution environment,
        including any necessary connections, worker pools, etc.
        """
        pass

    @abstractmethod
    def submit_task(self, job_index: int) -> Any:
        """Submit a single task and return a future.

        Args:
            job_index: The index of the time partition to process.

        Returns:
            A future object representing the pending task result.
        """
        pass

    @abstractmethod
    def wait_for_results(self, futures: List[Any]) -> Any:
        """Wait for futures and yield results as they complete.

        Args:
            futures: List of future objects from submit_task.

        Yields:
            Tuples of (job_index, result, duration, peak_memory_mb).
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources (shutdown, clear, etc.).

        This method should properly shutdown the parallel execution
        environment and release any resources.
        """
        pass

    def run(self) -> List[Any]:
        """Main execution loop - common for all executors.

        This method orchestrates the parallel execution of DBClust tasks:
        1. Initialize the executor
        2. Build and shuffle the partition map
        3. Submit all tasks
        4. Process results with progress tracking
        5. Cleanup resources

        Returns:
            List of results from all tasks.
        """
        self.run_start_time = datetime.now()
        logger.info(f"Starting parallel execution with {self.name}")
        logger.info(f"Number of workers: {self.cfg.parallel.n_workers}")
        logger.info(f"Number of time partitions: {len(self.cfg.parallel.time_partitions)}")

        self.initialize()

        # Build partition map: job_index -> (start, end)
        indexed_partitions = list(enumerate(self.cfg.parallel.time_partitions))
        partition_map: Dict[int, Tuple] = {
            idx: (start, end) for idx, (start, end) in indexed_partitions
        }

        # Resume: skip already completed tasks
        done = self._load_completed()
        if done:
            indexed_partitions = [(idx, p) for idx, p in indexed_partitions if idx not in done]
            logger.info(f"Resuming: {len(done)} tasks already done, {len(indexed_partitions)} remaining")

        # Shuffle for load balancing
        random.shuffle(indexed_partitions)

        # Initialize CSV file for progress tracking (append mode when resuming)
        if not done:
            self._init_csv()

        # Submit tasks with a sliding window to avoid exhausting file descriptors
        # when n_tasks is large (e.g. 10k+).  We keep at most max_inflight futures
        # alive at any time; new ones are submitted as old ones complete.
        n_submit = len(indexed_partitions)
        n_workers = self.cfg.parallel.n_workers or 1
        oversubscription = getattr(self.cfg.parallel, "oversubscription_factor", 1) or 1
        max_inflight = n_workers * oversubscription
        logger.info(
            f"Submitting {n_submit} tasks (sliding window, max_inflight={max_inflight})..."
        )

        # Process results with progress tracking (windowed submission)
        results = self._process_results_windowed(
            indexed_partitions, partition_map, max_inflight
        )

        self.cleanup()
        self._write_execution_summary(results)
        logger.info(f"Parallel execution completed with {self.name}")

        return results

    @property
    def _completed_path(self) -> str:
        """Path to the completed-tasks checkpoint file."""
        return self.profile_csv_path.replace(".csv", ".completed.json")

    def _load_completed(self) -> Set[int]:
        """Load set of already-completed task indices from checkpoint file."""
        if not os.path.exists(self._completed_path):
            return set()
        try:
            with open(self._completed_path) as f:
                return set(json.load(f))
        except Exception:
            return set()

    def _mark_completed(self, job_index: int) -> None:
        """Append a task index to the checkpoint file."""
        done = self._load_completed()
        done.add(job_index)
        with open(self._completed_path, "w") as f:
            json.dump(list(done), f)

    def _init_csv(self) -> None:
        """Initialize the CSV file for progress tracking."""
        os.makedirs(os.path.dirname(self.profile_csv_path), exist_ok=True)
        with open(self.profile_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()

    def _process_results_windowed(
        self,
        indexed_partitions: List[Tuple],
        partition_map: Dict[int, Tuple],
        max_inflight: int,
    ) -> List[Any]:
        """Submit and collect tasks with a sliding window.

        Keeps at most *max_inflight* futures alive at any time so that the OS
        file-descriptor limit is never exhausted even with 10k+ total tasks.
        """
        from collections import deque

        total_tasks = len(indexed_partitions)
        submit_iter = iter(indexed_partitions)
        submitted = 0
        pending: deque = deque()
        completed_count = 0
        results = []
        log_every = max(1, total_tasks // 10)
        processing_start = datetime.now()

        def _fill():
            nonlocal submitted
            while len(pending) < max_inflight and submitted < total_tasks:
                idx, _ = next(submit_iter)
                pending.append(self.submit_task(idx))
                submitted += 1
                if submitted % log_every == 0 or submitted == total_tasks:
                    logger.info(
                        f"Submitted {submitted}/{total_tasks} tasks "
                        f"({submitted / total_tasks * 100:.0f}%)"
                    )

        _fill()

        # submitted_box wraps the counter in a list so _stream_results can
        # mutate it (integers are immutable in Python).
        submitted_box = [submitted]
        for job_index, result, duration, peak_memory_mb in self._stream_results(
            pending, submit_iter, submitted_box, total_tasks, log_every
        ):
            completed_count += 1
            progress_pct = (completed_count / total_tasks) * 100
            partition_start, partition_end = partition_map.get(job_index, (None, None))
            completion_time = datetime.now()
            task_start_time = datetime.fromtimestamp(completion_time.timestamp() - duration)

            with open(self.profile_csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                writer.writerow({
                    "task_index": job_index,
                    "start_time": task_start_time.isoformat(),
                    "completion_time": completion_time.isoformat(),
                    "duration_sec": f"{duration:.2f}",
                    "peak_memory_mb": f"{peak_memory_mb:.1f}" if peak_memory_mb else "N/A",
                    "completed_count": completed_count,
                    "total_tasks": total_tasks,
                    "progress_pct": f"{progress_pct:.1f}",
                    "time_partition_start": str(partition_start) if partition_start else "N/A",
                    "time_partition_end": str(partition_end) if partition_end else "N/A",
                })

            elapsed = (datetime.now() - processing_start).total_seconds()
            elapsed_str = f"{elapsed/3600:.1f}h" if elapsed > 3600 else f"{elapsed/60:.0f}min"
            rate = completed_count / elapsed if elapsed > 0 else 0
            remaining = total_tasks - completed_count
            eta_sec = remaining / rate if rate > 0 else 0
            eta_str = f"{eta_sec/3600:.1f}h" if eta_sec > 3600 else f"{eta_sec/60:.0f}min"
            msg = (
                f"[{completed_count}/{total_tasks}] ({progress_pct:.1f}%) "
                f"task {job_index} done in {duration:.0f}s "
                f"— elapsed {elapsed_str} — ETA {eta_str}"
            )
            logger.info(msg)
            print(msg, flush=True)
            if job_index >= 0:
                self._mark_completed(job_index)
            results.append(result)

        return results

    def _stream_results(self, pending, submit_iter, submitted_box, total_tasks, log_every):
        """Yield results as futures complete, refilling the window one-for-one.

        Uses as_completed.add() if the executor exposes self._as_completed,
        so all in-flight futures are processed in a single streaming pass with
        no artificial serialisation.

        submitted_box is a one-element list so the counter is passed by reference.
        """
        for job_index, result, duration, peak_memory_mb in self.wait_for_results(list(pending)):
            pending.clear()
            yield job_index, result, duration, peak_memory_mb
            # Submit one replacement and inject it into the live as_completed iterator
            if submitted_box[0] < total_tasks:
                try:
                    idx, _ = next(submit_iter)
                    new_future = self.submit_task(idx)
                    submitted_box[0] += 1
                    if submitted_box[0] % log_every == 0 or submitted_box[0] == total_tasks:
                        logger.info(
                            f"Submitted {submitted_box[0]}/{total_tasks} tasks "
                            f"({submitted_box[0] / total_tasks * 100:.0f}%)"
                        )
                    ac = getattr(self, "_as_completed", None)
                    pf = getattr(self, "_pending_futures", None)
                    if ac is not None:
                        ac.add(new_future)       # Dask: inject into live as_completed
                    elif pf is not None:
                        pf.append(new_future)    # Ray: inject into live ray.wait list
                    else:
                        pending.append(new_future)  # fallback
                except StopIteration:
                    pass

    def _process_results(
        self,
        futures: List[Any],
        future_to_index: Dict[int, int],
        partition_map: Dict[int, Tuple],
    ) -> List[Any]:
        """Process results with CSV progress tracking.

        Args:
            futures: List of future objects.
            future_to_index: Mapping from future id to job index.
            partition_map: Mapping from job index to (start, end) times.

        Returns:
            List of results from all tasks.
        """
        total_tasks = len(futures)
        completed_count = 0
        results = []
        processing_start = datetime.now()

        for job_index, result, duration, peak_memory_mb in self.wait_for_results(futures):
            completed_count += 1
            progress_pct = (completed_count / total_tasks) * 100

            # Get partition times
            partition_start, partition_end = partition_map.get(job_index, (None, None))

            # Calculate task start_time from completion_time - duration
            completion_time = datetime.now()
            task_start_time = datetime.fromtimestamp(
                completion_time.timestamp() - duration
            )

            # Write to CSV
            with open(self.profile_csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                writer.writerow({
                    "task_index": job_index,
                    "start_time": task_start_time.isoformat(),
                    "completion_time": completion_time.isoformat(),
                    "duration_sec": f"{duration:.2f}",
                    "peak_memory_mb": f"{peak_memory_mb:.1f}" if peak_memory_mb else "N/A",
                    "completed_count": completed_count,
                    "total_tasks": total_tasks,
                    "progress_pct": f"{progress_pct:.1f}",
                    "time_partition_start": str(partition_start) if partition_start else "N/A",
                    "time_partition_end": str(partition_end) if partition_end else "N/A",
                })

            elapsed = (datetime.now() - processing_start).total_seconds()
            elapsed_str = f"{elapsed/3600:.1f}h" if elapsed > 3600 else f"{elapsed/60:.0f}min"
            rate = completed_count / elapsed if elapsed > 0 else 0
            remaining = total_tasks - completed_count
            eta_sec = remaining / rate if rate > 0 else 0
            eta_str = f"{eta_sec/3600:.1f}h" if eta_sec > 3600 else f"{eta_sec/60:.0f}min"
            logger.info(
                f"[{completed_count}/{total_tasks}] ({progress_pct:.1f}%) "
                f"task {job_index} done in {duration:.0f}s "
                f"— elapsed {elapsed_str} — ETA {eta_str}"
            )
            if job_index >= 0:
                self._mark_completed(job_index)
            results.append(result)

        return results

    def _write_execution_summary(self, results: List[Any]) -> None:
        """Write the execution summary CSV file.

        Args:
            results: List of results from all completed tasks.
        """
        end_time = datetime.now()
        if self.run_start_time is None:
            self.run_start_time = end_time
        total_duration = (end_time - self.run_start_time).total_seconds()

        os.makedirs(os.path.dirname(self.summary_csv_path), exist_ok=True)
        with open(self.summary_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            writer.writerow(["executor", self.name])
            writer.writerow(["start_time", self.run_start_time.isoformat()])
            writer.writerow(["end_time", end_time.isoformat()])
            writer.writerow(["total_duration_sec", f"{total_duration:.2f}"])
            writer.writerow(["n_workers", self.cfg.parallel.n_workers])
            writer.writerow(["total_tasks", len(results)])

        logger.info(f"Execution summary written to {self.summary_csv_path}")
