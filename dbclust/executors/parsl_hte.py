"""Parsl HighThroughputExecutor implementation for DBClust.

This executor uses Parsl with HighThroughputExecutor for true multi-process
parallel execution on a single machine. It avoids GIL limitations and provides
better CPU utilization for CPU-bound tasks.
"""

import logging
import time
from concurrent.futures import as_completed
from typing import Any, Generator, List

import parsl
from parsl.app.app import python_app
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider

from dbclust.config import DBClustConfig
from dbclust.executors.base import ExecutorBase

logger = logging.getLogger("dbclust")


@python_app
def _run_dbclust_task(cfg, job_index):
    """Parsl python_app wrapper for dbclust task execution.

    This function is decorated with @python_app to enable parallel execution
    through Parsl HighThroughputExecutor. Each task runs in a separate process,
    avoiding GIL limitations.

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


class ParslHTEExecutor(ExecutorBase):
    """Parsl executor using HighThroughputExecutor with LocalProvider.

    This executor provides true multi-process parallelism, avoiding Python's
    GIL limitations. It is suitable for CPU-bound tasks and provides better
    utilization of multi-core systems.

    Attributes:
        cfg: DBClust configuration object.
    """

    @property
    def name(self) -> str:
        return "Parsl HighThroughputExecutor"

    def _get_provider(self):
        """Get the provider for HTE. Override in subclasses for different providers."""
        return LocalProvider(
            init_blocks=1,
            min_blocks=0,
            max_blocks=1,
        )

    def initialize(self) -> None:
        """Initialize Parsl with HighThroughputExecutor."""
        # Reduce Parsl logging noise - must be done BEFORE creating executor
        if hasattr(parsl, "set_stream_logger"):
            parsl.set_stream_logger(level=logging.WARNING)

        # Silence all parsl loggers including HTE subloggers
        for logger_name in [
            "parsl",
            "parsl.dataflow.dflow",
            "parsl.dataflow.memoization",
            "parsl.process_loggers",
            "parsl.jobs.strategy",
            "parsl.usage_tracking.usage",
            "parsl.executors.high_throughput.executor",
            "parsl.executors.high_throughput.interchange",
            "parsl.executors.high_throughput.manager",
            "parsl.executors.high_throughput.process_worker_pool",
            "parsl.executors.high_throughput.zmq_pipes",
            "parsl.executors.status_handling",
            "parsl.serialize.facade",
            "parsl.providers",
            "parsl.channels",
            "parsl.utils",
        ]:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

        # Configure HighThroughputExecutor for true multi-process parallelism
        executor = HighThroughputExecutor(
            label="dbclust_hte",
            max_workers_per_node=self.cfg.parallel.n_workers,
            cores_per_worker=1,
            provider=self._get_provider(),
        )

        config = Config(
            executors=[executor],
            run_dir=self.cfg.parallel._temp_dir if self.cfg.parallel._temp_dir else "runinfo",
            retries=3,
        )

        parsl.load(config)
        logger.info(
            f"Parsl HighThroughputExecutor initialized with {self.cfg.parallel.n_workers} workers"
        )

    def submit_task(self, job_index: int) -> Any:
        """Submit a DBClust task to Parsl HTE.

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
                yield (-1, False, 0, 0)

    def cleanup(self) -> None:
        """Cleanup Parsl resources."""
        try:
            parsl.dfk().cleanup()
            parsl.clear()
            logger.info("Parsl HighThroughputExecutor cleaned up")
        except Exception as e:
            logger.warning(f"Error during Parsl cleanup: {e}")
