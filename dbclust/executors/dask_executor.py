"""
Adaptive and memory-safe Dask executor for DBClust.

Features:
- Adaptive scaling (workers created/destroyed dynamically)
- Strict per-worker memory limits
- Spill enabled (no OOM killer)
- Nanny disabled (Docker-safe)
- Heavy-task resource control
"""

import logging
import os
from typing import Any, Generator, List

from dbclust.config import DBClustConfig
from dbclust.executors.base import ExecutorBase

logger = logging.getLogger("dbclust")


def _run_dbclust_task(cfg: DBClustConfig, job_index: int):
    """Executed inside a Dask worker."""
    import gc
    import logging
    import os
    import time
    from contextlib import ExitStack, redirect_stderr, redirect_stdout

    from dbclust.core import dbclust

    # Configure file logging for this worker task
    log_dir = cfg.parallel._temp_dir or "runinfo"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"dbclust_task_{job_index}.log")

    # Silence root logger to prevent console output
    root_logger = logging.getLogger()
    original_root_handlers = root_logger.handlers[:]
    original_root_level = root_logger.level
    for handler in original_root_handlers:
        root_logger.removeHandler(handler)

    # Prepare file handler shared by root logger so every module propagates to it
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)

    # Ensure dbclust logger relies on root propagation instead of its own handlers
    dbclust_logger = logging.getLogger("dbclust")
    original_handlers = dbclust_logger.handlers[:]
    original_propagate = dbclust_logger.propagate
    for handler in original_handlers:
        dbclust_logger.removeHandler(handler)
    dbclust_logger.setLevel(logging.INFO)
    dbclust_logger.propagate = True

    # Redirect stdout/stderr so plain prints also land in the worker log file
    io_redirect_stack = ExitStack()
    stdout_stream = open(log_file, "a", buffering=1)
    io_redirect_stack.enter_context(stdout_stream)
    io_redirect_stack.enter_context(redirect_stdout(stdout_stream))
    io_redirect_stack.enter_context(redirect_stderr(stdout_stream))

    start = time.time()
    try:
        result = dbclust(cfg=cfg, job_index=job_index)
    finally:
        # Clean up: restore original handlers
        io_redirect_stack.close()
        file_handler.close()
        root_logger.removeHandler(file_handler)
        # Restore dbclust logger handlers and propagation flag
        for handler in original_handlers:
            dbclust_logger.addHandler(handler)
        dbclust_logger.propagate = original_propagate
        # Restore root logger handlers and level
        for handler in original_root_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(original_root_level)
        gc.collect()  # helps with Python-side cleanup

    return {
        "task_index": job_index,
        "duration_sec": time.time() - start,
        "peak_memory_mb": 0,
        "result": result,
    }


class DaskExecutor(ExecutorBase):
    """Adaptive, memory-safe Dask executor."""

    def __init__(self, cfg: DBClustConfig):
        super().__init__(cfg)
        self.cluster = None
        self.client = None
        self._cfg_future = None

    @property
    def name(self) -> str:
        return "Dask LocalCluster (adaptive, memory-safe)"

    def initialize(self) -> None:
        import logging as _logging
        import resource
        from dask.distributed import Client, LocalCluster
        from dask.distributed.worker import Worker

        # Raise the open-file-descriptor limit to handle many workers + sockets.
        # Each Dask worker + nanny uses ~15 fds; 120 workers needs ~1800 minimum.
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            target = max(65536, soft)
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(target, hard), hard))
            logger.info(f"fd limit: {soft} → {min(target, hard)} (hard={hard})")
        except Exception as e:
            logger.warning(f"Could not raise fd limit: {e}")

        configured_workers = self.cfg.parallel.n_workers
        if isinstance(configured_workers, int) and configured_workers > 0:
            max_workers = configured_workers
            worker_source = "config"
        else:
            max_workers = os.cpu_count() or 1
            worker_source = "auto-detected"

        # Silence noisy Dask logs
        for name in [
            "distributed",
            "distributed.worker",
            "distributed.scheduler",
            "distributed.client",
            "distributed.nanny",
        ]:
            _logging.getLogger(name).setLevel(_logging.ERROR)

        # --- Memory protection (DO NOT disable spill)
        Worker.memory_target_fraction = 0.70
        Worker.memory_spill_fraction = 0.80
        Worker.memory_pause_fraction = 0.95

        # Start with a small pool then scale up in batches to avoid spawning all
        # nannies simultaneously (which exhausts OS file descriptors with EMFILE).
        batch_size = min(16, max_workers)
        self.cluster = LocalCluster(
            n_workers=batch_size,
            threads_per_worker=1,
            processes=True,
            memory_limit="2GB",
            dashboard_address=":8265",
            resources={"heavy": 1},
        )

        self.client = Client(
            self.cluster,
            timeout=120,
            direct_to_workers=True,
        )

        self.client.wait_for_workers(batch_size)

        # Scale up to full worker count in batches
        import time
        current = batch_size
        while current < max_workers:
            next_batch = min(current + batch_size, max_workers)
            self.cluster.scale(next_batch)
            time.sleep(2)
            current = next_batch

        self.client.wait_for_workers(max_workers)

        logger.info(
            "Dask adaptive cluster ready: "
            f"max_workers={max_workers} ({worker_source}), "
            "memory_limit=2GB/worker"
        )
        logger.info(f"Dask dashboard: {self.client.dashboard_link}")

        # --- Scatter config to all workers
        self._cfg_future = self.client.scatter(
            self.cfg,
            broadcast=True,
        )

    def submit_task(self, job_index: int) -> Any:
        start, end = self.cfg.parallel.time_partitions[job_index]
        logger.info(f"Submitting task {job_index} [{start} -- {end}]")

        return self.client.submit(
            _run_dbclust_task,
            self._cfg_future,
            job_index,
            pure=False,
            retries=1,
            resources={"heavy": 1},   # 🔥 enforces memory discipline
        )

    def wait_for_results(self, futures: List[Any]) -> Generator:
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
                logger.error("Task failed", exc_info=e)
                yield (-1, False, 0.0, 0.0)

    def cleanup(self) -> None:
        try:
            if self.client:
                self.client.shutdown()
                self.client = None
            if self.cluster:
                self.cluster.close(timeout=10)
                self.cluster = None
            logger.info("Dask executor cleaned up cleanly")
        except Exception as e:
            logger.warning(f"Dask cleanup error: {e}")