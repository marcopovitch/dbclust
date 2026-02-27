"""Abstract base class for all DBClust executors.

This module defines the ExecutorBase class that all parallel execution
backends must inherit from.
"""

import csv
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

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

        # Shuffle for load balancing
        random.shuffle(indexed_partitions)

        # Initialize CSV file for progress tracking
        self._init_csv()

        # Submit all tasks
        n_submit = len(indexed_partitions)
        logger.info(f"Submitting {n_submit} tasks...")
        futures = []
        future_to_index = {}
        log_every = max(1, n_submit // 10)
        for i, (idx, (start, end)) in enumerate(indexed_partitions):
            future = self.submit_task(idx)
            futures.append(future)
            future_to_index[id(future)] = idx
            if (i + 1) % log_every == 0 or (i + 1) == n_submit:
                logger.info(f"Submitted {i + 1}/{n_submit} tasks ({(i+1)/n_submit*100:.0f}%)")

        # Process results with progress tracking
        results = self._process_results(futures, future_to_index, partition_map)

        self.cleanup()
        self._write_execution_summary(results)
        logger.info(f"Parallel execution completed with {self.name}")

        return results

    def _init_csv(self) -> None:
        """Initialize the CSV file for progress tracking."""
        os.makedirs(os.path.dirname(self.profile_csv_path), exist_ok=True)
        with open(self.profile_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()

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
