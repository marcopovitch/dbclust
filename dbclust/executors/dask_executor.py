"""
Memory-safe Dask executor for DBClust.

Strategy for I/O-bound tasks (NLLoc subprocess, ~80% wait time):
- n_workers = configured_workers * oversubscription_factor separate processes
- threads_per_worker = 1  (no GIL contention, clean process isolation)
- Workers started in batches to avoid EMFILE (too many open files)
- Futures submitted via sliding window so we never hold 10k+ open connections
"""

import logging
import os
from typing import Any, Generator, List

from dbclust.config import DBClustConfig
from dbclust.executors.base import ExecutorBase

logger = logging.getLogger("dbclust")


def _run_dbclust_task(cfg: DBClustConfig, job_index: int):
    """Executed inside a Dask worker process."""
    import gc
    import logging
    import os
    import time
    from contextlib import ExitStack, redirect_stderr, redirect_stdout

    from dbclust.core import dbclust

    log_dir = cfg.parallel._temp_dir or "runinfo"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"dbclust_task_{job_index}.log")

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

    dbclust_logger = logging.getLogger("dbclust")
    original_handlers = dbclust_logger.handlers[:]
    original_propagate = dbclust_logger.propagate
    for handler in original_handlers:
        dbclust_logger.removeHandler(handler)
    dbclust_logger.setLevel(logging.INFO)
    dbclust_logger.propagate = True

    io_redirect_stack = ExitStack()
    stdout_stream = open(log_file, "a", buffering=1)
    io_redirect_stack.enter_context(stdout_stream)
    io_redirect_stack.enter_context(redirect_stdout(stdout_stream))
    io_redirect_stack.enter_context(redirect_stderr(stdout_stream))

    start = time.time()
    try:
        result = dbclust(cfg=cfg, job_index=job_index)
    finally:
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

    return {
        "task_index": job_index,
        "duration_sec": time.time() - start,
        "peak_memory_mb": 0,
        "result": result,
    }


class DaskExecutor(ExecutorBase):
    """Memory-safe Dask executor optimised for I/O-bound tasks."""

    def __init__(self, cfg: DBClustConfig):
        super().__init__(cfg)
        self.cluster = None
        self.client = None
        self._cfg_future = None

    @property
    def name(self) -> str:
        return "Dask LocalCluster (memory-safe)"

    def initialize(self) -> None:
        import logging as _logging
        import resource
        import time
        from dask.distributed import Client, LocalCluster
        from dask.distributed.worker import Worker

        # Raise the fd limit — each worker+nanny uses ~15 fds.
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            target = max(65536, soft)
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(target, hard), hard))
            logger.info(f"fd limit: {soft} → {min(target, hard)} (hard={hard})")
        except Exception as e:
            logger.warning(f"Could not raise fd limit: {e}")

        configured_workers = self.cfg.parallel.n_workers or os.cpu_count() or 1
        oversubscription = getattr(self.cfg.parallel, "oversubscription_factor", 1) or 1

        # Tasks are I/O-bound (NLLoc subprocess).  Use one process per logical
        # "slot" so they truly run in parallel without GIL interference.
        # This mirrors Ray's  num_cpus = 1 / oversubscription_factor.
        max_workers = configured_workers * oversubscription

        # Silence noisy Dask/Bokeh/Tornado logs
        for name in [
            "distributed",
            "distributed.worker",
            "distributed.scheduler",
            "distributed.client",
            "distributed.nanny",
            "tornado",
            "tornado.access",
            "tornado.application",
            "tornado.general",
            "bokeh",
            "asyncio",
        ]:
            _logging.getLogger(name).setLevel(_logging.CRITICAL)

        # Memory protection
        Worker.memory_target_fraction = 0.70
        Worker.memory_spill_fraction = 0.80
        Worker.memory_pause_fraction = 0.95

        # Start in batches to avoid spawning all nannies simultaneously (EMFILE).
        batch_size = min(16, max_workers)
        self.cluster = LocalCluster(
            n_workers=batch_size,
            threads_per_worker=1,   # one task per process — no GIL issues
            processes=True,
            memory_limit="2GB",
            dashboard_address=":8265",
        )

        self.client = Client(self.cluster, timeout=120, direct_to_workers=True)
        self.client.wait_for_workers(batch_size)

        current = batch_size
        while current < max_workers:
            next_count = min(current + batch_size, max_workers)
            self.cluster.scale(next_count)
            time.sleep(2)
            current = next_count

        self.client.wait_for_workers(max_workers)

        # Diagnostic: log actual scheduler state
        info = self.client.scheduler_info()
        actual_workers = len(info.get("workers", {}))
        logger.info(
            f"Dask cluster ready: {actual_workers}/{max_workers} workers up "
            f"({configured_workers} CPUs × {oversubscription}x oversubscription), "
            "memory_limit=2GB/worker"
        )
        logger.info(f"Dask dashboard: {self.client.dashboard_link}")

        self._cfg_future = self.client.scatter(self.cfg, broadcast=True)

    def submit_task(self, job_index: int) -> Any:
        return self.client.submit(
            _run_dbclust_task,
            self._cfg_future,
            job_index,
            pure=False,
            retries=1,
        )

    def wait_for_results(self, futures: List[Any]) -> Generator:
        from dask.distributed import as_completed

        logger.info(f"as_completed starting on {len(futures)} futures")
        ac = as_completed(futures)
        self._as_completed = ac
        for future in ac:
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
        self._as_completed = None

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
