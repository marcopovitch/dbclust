"""Ray executor implementation for DBClust.

This executor uses Ray for distributed parallel execution. Ray provides
excellent scalability and can run on a single machine or across a cluster.
"""

import logging
import os
import time
from typing import Any, Generator, List

import ray

from dbclust.config import DBClustConfig
from dbclust.executors.base import ExecutorBase

logger = logging.getLogger("dbclust")


# Ray task must be defined at module level
@ray.remote(max_calls=1, max_retries=5, num_cpus=1, memory=5 * 1024**3)  # 5 GB
def _run_dbclust_task(cfg, job_index):
    """Ray remote function for dbclust task execution.

    This function is decorated with @ray.remote to enable parallel execution
    through Ray. Each task runs in a separate process.

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


class RayExecutor(ExecutorBase):
    """Ray executor for distributed parallel execution.

    This executor uses Ray for parallel task execution. It supports both
    local execution and distributed clusters.

    Attributes:
        cfg: DBClust configuration object.
        context: Ray context after initialization.
    """

    def __init__(self, cfg: DBClustConfig):
        super().__init__(cfg)
        self.context = None

    @property
    def name(self) -> str:
        return "Ray"

    def initialize(self) -> None:
        """Initialize Ray cluster."""
        # Set Ray environment variables
        os.environ["RAY_DEDUP_LOGS"] = "0"
        os.environ["RAY_COLOR_PREFIX"] = "1"
        os.environ["RAY_enable_oom_killer"] = "1"
        os.environ["RAY_memory_usage_threshold"] = "0.95"

        self.context = ray.init(
            num_cpus=self.cfg.parallel.n_workers,
            _temp_dir=self.cfg.parallel._temp_dir,
            dashboard_host="0.0.0.0",
            dashboard_port=8265,
            include_dashboard=True,
        )
        logger.info(f"Ray initialized with {self.cfg.parallel.n_workers} CPUs")
        logger.info(f"Dashboard URL: {self.context.dashboard_url}")

    def submit_task(self, job_index: int) -> Any:
        """Submit a DBClust task to Ray.

        Args:
            job_index: Index of the time partition to process.

        Returns:
            Ray ObjectRef representing the pending task.
        """
        start, end = self.cfg.parallel.time_partitions[job_index]
        logger.info(f"Submitting task {job_index} [{start} -- {end}]")
        return _run_dbclust_task.remote(self.cfg, job_index)

    def wait_for_results(self, futures: List[Any]) -> Generator:
        """Wait for Ray futures and yield results as they complete.

        Uses ray.wait() to process tasks as they complete for progressive
        memory release.

        Args:
            futures: List of Ray ObjectRef objects.

        Yields:
            Tuples of (job_index, result, duration, peak_memory_mb).
        """
        remaining_futures = list(futures)
        while remaining_futures:
            done, remaining_futures = ray.wait(remaining_futures, num_returns=1)
            for future in done:
                try:
                    r = ray.get(future)
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
        """Shutdown Ray cluster."""
        try:
            ray.shutdown()
            logger.info("Ray executor cleaned up")
        except Exception as e:
            logger.warning(f"Error during Ray shutdown: {e}")
