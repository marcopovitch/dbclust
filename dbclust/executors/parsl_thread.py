"""
Parsl ThreadPoolExecutor implementation for DBClust.

This executor uses Parsl with ThreadPoolExecutor for parallel execution
on a single machine. It is suitable for I/O-bound tasks but may be limited
by Python's GIL for CPU-bound tasks.
"""

import logging
import os
from concurrent.futures import as_completed
from typing import Any, Dict, Generator, List

import parsl
from parsl.app.app import python_app
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor

from dbclust.config import DBClustConfig
from dbclust.executors.base import ExecutorBase

logger = logging.getLogger("dbclust")


@python_app
def _run_dbclust_task(cfg: DBClustConfig, job_index: int) -> Dict:
    """Parsl python_app wrapper for dbclust task execution.

    This function is decorated with @python_app to enable parallel execution
    through Parsl. It imports dbclust from core to avoid serialization issues.

    Args:
        cfg: DBClustConfig object with all parameters.
        job_index: Index of the time partition to process.

    Returns:
        Dictionary with task_index, duration_sec, peak_memory_mb, and result.
    """
    import logging
    import os
    import threading
    import time

    from dbclust.core import dbclust

    # Configure logging to file
    log_dir = cfg.parallel._temp_dir if cfg.parallel._temp_dir else "runinfo"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"dbclust_task_{job_index}.log")

    # Create a filter that only allows logs from the current thread
    current_thread = threading.current_thread()

    class ThreadFilter(logging.Filter):
        def filter(self, record):
            return threading.current_thread() == current_thread

    # Create file handler with thread filter
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    file_handler.setLevel(logging.INFO)
    file_handler.addFilter(ThreadFilter())

    # Create a filter to block current thread logs from existing handlers
    class BlockThreadFilter(logging.Filter):
        def filter(self, record):
            return threading.current_thread() != current_thread

    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    # Add blocking filter to existing handlers to prevent console output
    existing_handlers = root_logger.handlers[:]
    for handler in existing_handlers:
        if handler != file_handler:
            handler.addFilter(BlockThreadFilter())

    start_time = time.time()
    try:
        result = dbclust(cfg=cfg, job_index=job_index)
    finally:
        # Clean up: remove blocking filters from existing handlers
        for handler in existing_handlers:
            if handler != file_handler:
                for f in list(handler.filters):
                    if isinstance(f, BlockThreadFilter):
                        handler.removeFilter(f)
        # Clean up file handler
        file_handler.close()
        root_logger.removeHandler(file_handler)
        root_logger.setLevel(original_level)

    duration_sec = time.time() - start_time

    return {
        "task_index": job_index,
        "duration_sec": duration_sec,
        "peak_memory_mb": 0,
        "result": result,
    }


class ParslThreadExecutor(ExecutorBase):
    """Parsl executor using ThreadPoolExecutor.

    This executor is the default for local parallel execution. It uses
    Python threads which work well for I/O-bound tasks but may be limited
    by the GIL for CPU-intensive processing.
    
    At least it works on macOS.

    Attributes:
        cfg: DBClust configuration object.
    """

    @property
    def name(self) -> str:
        return "Parsl ThreadPoolExecutor"

    def initialize(self) -> None:
        """Initialize Parsl with ThreadPoolExecutor."""
        # Reduce Parsl logging noise
        if hasattr(parsl, "set_stream_logger"):
            parsl.set_stream_logger(level=logging.WARNING)

        configured_workers = self.cfg.parallel.n_workers
        if isinstance(configured_workers, int) and configured_workers > 0:
            max_threads = configured_workers
            worker_source = "config"
        else:
            max_threads = os.cpu_count() or 1
            worker_source = "auto-detected"

        for logger_name in [
            "parsl",
            "parsl.dataflow.dflow",
            "parsl.dataflow.memoization",
            "parsl.process_loggers",
            "parsl.jobs.strategy",
            "parsl.usage_tracking.usage",
        ]:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

        # Configure ThreadPoolExecutor
        executor = ThreadPoolExecutor(
            label="dbclust_executor",
            max_threads=max_threads,
        )

        config = Config(
            executors=[executor],
            run_dir=self.cfg.parallel._temp_dir if self.cfg.parallel._temp_dir else "runinfo",
            retries=3,
        )

        parsl.load(config)
        logger.info(
            "Parsl ThreadPoolExecutor initialized with "
            f"{max_threads} threads ({worker_source})"
        )

    def submit_task(self, job_index: int) -> Any:
        """Submit a DBClust task to Parsl.

        Args:
            job_index: Index of the time partition to process.

        Returns:
            Parsl AppFuture representing the pending task.
        """
        start, end = self.cfg.parallel.time_partitions[job_index]
        logger.info(f"Submitting task {job_index} [{start} -- {end}]")
        return _run_dbclust_task(self.cfg, job_index)

    def wait_for_results(self, futures: List[Any]) -> Generator:
        """Wait for Parsl futures and yield results as they complete.

        Args:
            futures: List of Parsl AppFuture objects.

        Yields:
            Tuples of (job_index, result, duration, peak_memory_mb).
        """
        for completed_future in as_completed(futures):
            try:
                r = completed_future.result()
                yield (
                    r["task_index"],
                    r["result"],
                    r["duration_sec"],
                    r["peak_memory_mb"],
                )
            except Exception as e:
                logger.error(f"Task failed with error: {e}")
                # Yield a failure result
                yield (-1, False, 0, 0)

    def cleanup(self) -> None:
        """Cleanup Parsl resources."""
        try:
            parsl.dfk().cleanup()
            parsl.clear()
            logger.info("Parsl ThreadPoolExecutor cleaned up")
        except Exception as e:
            logger.warning(f"Error during Parsl cleanup: {e}")
