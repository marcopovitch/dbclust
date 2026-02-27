"""Ray executor implementation for DBClust.

This executor uses Ray for distributed parallel execution. Ray provides
excellent scalability and can run on a single machine or across a cluster.
"""

import logging
import os
import time
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from typing import Any, Generator, List

import ray

from dbclust.config import DBClustConfig
from dbclust.executors.base import ExecutorBase

logger = logging.getLogger("dbclust")


# Ray task must be defined at module level
# max_calls used to be pinned to 1 to guard against potential memory leaks,
# but keeping workers alive is required for full CPU utilization.
@ray.remote(max_retries=5, num_cpus=1, memory=5 * 1024**3)  # 5 GB
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
    import gc
    import logging
    import os
    import time

    from dbclust.core import dbclust

    # Configure file logging for this worker task
    log_dir = cfg.parallel._temp_dir if cfg.parallel._temp_dir else "runinfo"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"dbclust_task_{job_index}.log")

    # Silence root logger and attach a dedicated file handler
    root_logger = logging.getLogger()
    original_root_handlers = root_logger.handlers[:]
    original_root_level = root_logger.level
    for handler in original_root_handlers:
        root_logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)

    # Ensure dbclust logger forwards to root
    dbclust_logger = logging.getLogger("dbclust")
    original_handlers = dbclust_logger.handlers[:]
    original_propagate = dbclust_logger.propagate
    for handler in original_handlers:
        dbclust_logger.removeHandler(handler)
    dbclust_logger.setLevel(logging.INFO)
    dbclust_logger.propagate = True

    # Redirect stdout/stderr so prints also land in the log file
    io_redirect_stack = ExitStack()
    stdout_stream = open(log_file, "a", buffering=1)
    io_redirect_stack.enter_context(stdout_stream)
    io_redirect_stack.enter_context(redirect_stdout(stdout_stream))
    io_redirect_stack.enter_context(redirect_stderr(stdout_stream))

    start_time = time.time()
    try:
        result = dbclust(cfg=cfg, job_index=job_index)
    finally:
        # Clean up: restore original handlers
        io_redirect_stack.close()
        file_handler.close()
        root_logger.removeHandler(file_handler)
        for handler in original_handlers:
            dbclust_logger.addHandler(handler)
        dbclust_logger.propagate = original_propagate
        for handler in original_root_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(original_root_level)
        gc.collect()

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

        configured_workers = self.cfg.parallel.n_workers
        if isinstance(configured_workers, int) and configured_workers > 0:
            total_cpus = configured_workers
            cpu_source = "config"
        else:
            total_cpus = os.cpu_count() or 1
            cpu_source = "auto-detected"

        # Ray uses _temp_dir for AF_UNIX sockets which have a 107-byte path limit.
        # Use /tmp/ray to keep socket paths short regardless of the configured temp dir.
        self.context = ray.init(
            num_cpus=total_cpus,
            _temp_dir="/tmp/ray",
            dashboard_host="0.0.0.0",
            dashboard_port=8265,
            include_dashboard=True,
        )
        logger.info(f"Ray initialized with {total_cpus} CPUs ({cpu_source})")
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
