"""Dask executor implementation for DBClust.

This executor uses Dask LocalCluster for parallel execution on a single
machine with multiple processes.
"""

import logging
import multiprocessing as mp
import time
from typing import Any, Generator, List

from dbclust.config import DBClustConfig
from dbclust.executors.base import ExecutorBase

logger = logging.getLogger("dbclust")


def _run_dbclust_task(cfg, job_index):
    """Task function for Dask execution.

    This function is submitted to Dask workers for parallel execution.

    Args:
        cfg: DBClustConfig object with all parameters.
        job_index: Index of the time partition to process.

    Returns:
        Dictionary with task_index, duration_sec, peak_memory_mb, and result.
    """
    import time
    from dbclust.core import dbclust

    start_time = time.time()
    result = dbclust(cfg=cfg, job_index=job_index)
    end_time = time.time()

    duration_sec = end_time - start_time
    peak_memory_mb = 0  # Memory profiling disabled by default

    return {
        "task_index": job_index,
        "duration_sec": duration_sec,
        "peak_memory_mb": peak_memory_mb,
        "result": result,
    }


class DaskExecutor(ExecutorBase):
    """Dask executor using LocalCluster.

    This executor uses Dask's LocalCluster for parallel execution with
    separate processes. It provides a web dashboard for monitoring.

    Attributes:
        cfg: DBClust configuration object.
        cluster: Dask LocalCluster instance.
        client: Dask Client instance.
    """

    def __init__(self, cfg: DBClustConfig):
        super().__init__(cfg)
        self.cluster = None
        self.client = None

    @property
    def name(self) -> str:
        return "Dask LocalCluster"

    def initialize(self) -> None:
        """Initialize Dask LocalCluster and Client."""
        from dask.distributed import Client, LocalCluster
        from dask.distributed.worker import Worker

        # Set spawn method for multiprocessing
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # Already set

        # Set memory limits for workers
        Worker.memory_target_fraction = 0.8
        Worker.memory_spill_fraction = 0.9
        Worker.memory_pause_fraction = 0.95

        # Create LocalCluster
        self.cluster = LocalCluster(
            n_workers=self.cfg.parallel.n_workers,
            threads_per_worker=1,
            processes=True,
            memory_limit="auto",
            local_directory=self.cfg.parallel._temp_dir,
            dashboard_address=":8265",
        )

        # Create Client
        self.client = Client(
            self.cluster,
            timeout=30,
            direct_to_workers=True,
        )

        logger.info(
            f"Dask LocalCluster initialized with {self.cfg.parallel.n_workers} workers"
        )
        logger.info(f"Dask Dashboard URL: {self.client.dashboard_link}")

    def submit_task(self, job_index: int) -> Any:
        """Submit a DBClust task to Dask.

        Args:
            job_index: Index of the time partition to process.

        Returns:
            Dask Future representing the pending task.
        """
        start, end = self.cfg.parallel.time_partitions[job_index]
        logger.info(f"Submitting task {job_index} [{start} -- {end}]")
        return self.client.submit(_run_dbclust_task, self.cfg, job_index)

    def wait_for_results(self, futures: List[Any]) -> Generator:
        """Wait for Dask futures and yield results as they complete.

        Args:
            futures: List of Dask Future objects.

        Yields:
            Tuples of (job_index, result, duration, peak_memory_mb).
        """
        from dask.distributed import as_completed

        for future in as_completed(futures):
            try:
                r = future.result()
                yield (
                    r["task_index"],
                    r["result"],
                    r["duration_sec"],
                    r["peak_memory_mb"],
                )
            except Exception as e:
                logger.error(f"Task failed with error: {e}")
                yield (-1, False, 0, 0)

    def cleanup(self) -> None:
        """Cleanup Dask resources."""
        try:
            if self.client:
                self.client.close()
            if self.cluster:
                self.cluster.close()
            logger.info("Dask executor cleaned up")
        except Exception as e:
            logger.warning(f"Error during Dask cleanup: {e}")
