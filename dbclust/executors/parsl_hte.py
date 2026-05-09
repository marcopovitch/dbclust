"""Parsl HighThroughputExecutor implementation for DBClust.

This executor uses Parsl with HighThroughputExecutor for true multi-process
parallel execution on a single machine. It avoids GIL limitations and provides
better CPU utilization for CPU-bound tasks.
"""

import logging
import os
import random
from concurrent.futures import as_completed
from datetime import datetime
from typing import Any, Generator, List, Dict

import parsl
from parsl.app.app import python_app
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider

from dbclust.executors.base import ExecutorBase

logger = logging.getLogger("dbclust")


@python_app
def _run_dbclust_task(cfg_file: str, log_level: int, job_index: int) -> Dict:
    """Parsl python_app wrapper for dbclust task execution.

    Accepts the YAML config path instead of a DBClustConfig object to avoid
    serializing large objects (StationXML inventories, DataFrames) over ZMQ
    for every task. DBClustConfig is reconstructed once per worker process.

    Args:
        cfg_file: Path to the YAML config file.
        log_level: Logging level integer.
        job_index: Index of the time partition to process.

    Returns:
        Dictionary with task_index, duration_sec, peak_memory_mb, and result.
    """
    import logging
    import os
    import platform
    import socket
    import time

    from dbclust.config import DBClustConfig
    from dbclust.core import dbclust

    cfg = DBClustConfig(cfg_file)
    cfg.log_level = log_level

    log_dir = cfg.parallel._temp_dir if cfg.parallel._temp_dir else "runinfo"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"dbclust_task_{job_index}.log")

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    file_handler.setLevel(log_level)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)

    logging.getLogger("dbclust").info(
        f"Running on host={socket.gethostname()} "
        f"arch={platform.machine()} "
        f"processor={platform.processor()}"
    )

    start_time = time.time()
    try:
        result = dbclust(cfg=cfg, job_index=job_index)
    finally:
        file_handler.close()
        root_logger.removeHandler(file_handler)

    return {
        "task_index": job_index,
        "duration_sec": time.time() - start_time,
        "peak_memory_mb": 0,
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
        n_blocks = getattr(self, "_n_blocks", 1)
        return LocalProvider(
            init_blocks=n_blocks,
            min_blocks=n_blocks,
            max_blocks=n_blocks,
        )

    def initialize(self) -> None:
        """Initialize Parsl with HighThroughputExecutor."""
        import signal
        import psutil

        # Kill any orphaned process_worker_pool processes from a previous crashed run.
        # These hold ZMQ ports open and cause SIGSEGV (exit code -11) on the next launch.
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = " ".join(proc.info["cmdline"] or [])
                if "process_worker_pool" in cmdline:
                    logger.warning(f"Killing orphaned Parsl worker (pid {proc.pid})")
                    proc.send_signal(signal.SIGTERM)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Clean up any leftover Parsl state from a previous crashed run.
        # Without this, stale ZMQ sockets/ports can cause SIGSEGV (exit code -11)
        # when launching a new block.
        try:
            parsl.dfk().cleanup()
            parsl.clear()
            logger.debug("Cleaned up existing Parsl DFK before re-initializing")
        except Exception:
            pass  # No active DFK, nothing to clean

        # Suppress Parsl KeyError spam on stale job IDs from previous runs.
        # parsl.utils logs these at ERROR level but explicitly says "proceeding anyway" — harmless.
        logging.getLogger("parsl.utils").setLevel(logging.CRITICAL)
        logging.getLogger("parsl.providers.slurm.slurm").setLevel(logging.CRITICAL)

        # Reduce Parsl logging noise - must be done BEFORE creating executor
        if hasattr(parsl, "set_stream_logger"):
            parsl.set_stream_logger(level=logging.WARNING)

        configured_workers = self.cfg.parallel.n_workers
        if isinstance(configured_workers, int) and configured_workers > 0:
            max_workers = configured_workers
            worker_source = "config"
        else:
            max_workers = os.cpu_count() or 1
            worker_source = "auto-detected"

        # Limit workers to number of tasks to avoid idle workers
        n_tasks = len(self.cfg.parallel.time_partitions)
        if max_workers > n_tasks:
            logger.info(
                f"Reducing workers from {max_workers} to {n_tasks} (number of tasks)"
            )
            max_workers = n_tasks

        # Each dbclust worker spends ~80% of its time waiting for NLLoc subprocess.
        # Oversubscribe so that idle-waiting workers don't leave CPUs unused.
        # This is equivalent to Ray's num_cpus=1/oversubscription_factor per task.
        cpu_count = os.cpu_count() or 1
        oversubscription_factor = self.cfg.parallel.oversubscription_factor
        oversubscribed_workers = min(max_workers * oversubscription_factor, n_tasks)
        logger.info(
            f"Oversubscribing: {max_workers} logical → {oversubscribed_workers} workers "
            f"(factor {oversubscription_factor}x, {cpu_count} physical CPUs)"
        )
        max_workers = oversubscribed_workers

        # Use multiple blocks (process_worker_pool processes) for better parallelism.
        # Each block manages workers_per_block workers independently via its own ZMQ manager.
        workers_per_block = 16  # tunable: 8-32 is a good range
        n_blocks = max(1, max_workers // workers_per_block)
        workers_per_block = max_workers // n_blocks  # rebalance evenly
        self._n_blocks = n_blocks
        logger.info(
            f"Using {n_blocks} blocks × {workers_per_block} workers/block"
        )

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
        run_dir = self.cfg.parallel._temp_dir if self.cfg.parallel._temp_dir else "runinfo"
        executor = HighThroughputExecutor(
            label="dbclust_hte",
            max_workers_per_node=workers_per_block,
            cores_per_worker=1,
            provider=self._get_provider(),
            worker_debug=False,
            worker_logdir_root=run_dir,
            poll_period=100,  # ms, reduce polling overhead
            heartbeat_threshold=600,  # s, allow 10 min without heartbeat (default 120s)
            heartbeat_period=30,      # s, heartbeat frequency
        )

        config = Config(
            executors=[executor],
            run_dir=self.cfg.parallel._temp_dir if self.cfg.parallel._temp_dir else "runinfo",
            retries=0,  # no silent retries — propagate exceptions immediately for visibility
            strategy="none",  # disable auto scale-in which causes ZMQError mid-run
        )

        parsl.load(config)
        logger.info(
            "Parsl HighThroughputExecutor initialized with "
            f"{max_workers} workers ({worker_source})"
        )

    def submit_task(self, job_index: int) -> Any:
        """Submit a DBClust task to Parsl HTE.

        Args:
            job_index: Index of the time partition to process.

        Returns:
            Parsl AppFuture representing the pending task.
        """
        start, end = self.cfg.parallel.time_partitions[job_index]
        logger.debug(f"Submitting task {job_index} [{start} -- {end}]")
        return _run_dbclust_task(self.cfg.filename, self.cfg.log_level, job_index)

    def wait_for_results(self, futures: List[Any]) -> Generator:
        """Wait for Parsl futures and yield results as they complete.

        Args:
            futures: List of Parsl AppFuture objects.

        Yields:
            Tuples of (job_index, result, duration, peak_memory_mb).
        """
        from parsl.executors.high_throughput.errors import ManagerLost, WorkerLost

        future_to_index = getattr(self, "_future_to_index", {})

        for completed_future in as_completed(futures):
            job_index = future_to_index.get(completed_future, -1)
            try:
                r = completed_future.result()
                yield (
                    r["task_index"],
                    r["result"],
                    r["duration_sec"],
                    r["peak_memory_mb"],
                )
            except (ManagerLost, WorkerLost) as e:
                logger.error(f"Parsl worker/manager lost, task {job_index} will be skipped: {e}")
                yield (job_index, False, 0, 0)
            except Exception as e:
                import traceback
                logger.error(f"Task {job_index} failed with error: {e}\n{traceback.format_exc()}")
                yield (job_index, False, 0, 0)

    def run(self) -> List[Any]:
        """Override run() to submit ALL tasks upfront before collecting results.

        The base class uses a sliding window which causes Parsl to think execution
        is finished after the first batch completes (it shuts down its internal
        thread pool), leading to RuntimeError on subsequent callbacks.

        Parsl has its own internal queue/scheduler, so submitting all futures at
        once is the correct pattern: workers are throttled by max_workers_per_node,
        not by the number of submitted futures.
        """
        import csv
        import signal

        from dbclust.core import CSV_FIELDNAMES

        self.run_start_time = datetime.now()
        logger.info(f"Starting parallel execution with {self.name}")
        logger.info(f"Number of workers: {self.cfg.parallel.n_workers}")
        logger.info(f"Number of time partitions: {len(self.cfg.parallel.time_partitions)}")

        # Compute remaining tasks before initialize() so subclasses (e.g. ParslSlurmExecutor)
        # can cap the number of Slurm blocks to the actual workload.
        _all_partitions = list(enumerate(self.cfg.parallel.time_partitions or []))
        _done_preview = self._load_completed()
        self._n_remaining_tasks = len(_all_partitions) - len(_done_preview)
        logger.info(f"Remaining tasks before initialize(): {self._n_remaining_tasks}")

        self.initialize()
        if self.on_initialized:
            self.on_initialized()

        # Install signal handlers so Ctrl-C / SIGTERM triggers a clean Parsl shutdown.
        # IMPORTANT: the handler must NOT call cleanup() directly — it is invoked from a
        # thread (inside threading.Condition.wait) and Parsl's ZMQ calls are not
        # re-entrant / thread-safe.  We only set a flag here and raise KeyboardInterrupt
        # to unblock as_completed(); the actual cleanup happens in the finally block
        # below, which runs in the main thread.
        _interrupted = [False]
        _orig_sigint = signal.getsignal(signal.SIGINT)
        _orig_sigterm = signal.getsignal(signal.SIGTERM)

        def _handle_signal(signum, frame):
            if not _interrupted[0]:
                _interrupted[0] = True
                print(
                    f"\n[dbclust] Signal {signum} received — shutting down workers...",
                    flush=True,
                )
            # Raise KeyboardInterrupt to unblock as_completed() in the main thread.
            raise KeyboardInterrupt()

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        results: List[Any] = []
        try:
            indexed_partitions = list(enumerate(self.cfg.parallel.time_partitions or []))
            partition_map = {idx: (s, e) for idx, (s, e) in indexed_partitions}

            done = self._load_completed()
            already_done = len(done)
            total_overall = len(indexed_partitions)
            if done:
                indexed_partitions = [(idx, p) for idx, p in indexed_partitions if idx not in done]
                logger.info(
                    f"Resuming: {already_done} tasks already done, {len(indexed_partitions)} remaining"
                )

            random.shuffle(indexed_partitions)

            if not done:
                self._init_csv()

            total_tasks = len(indexed_partitions)

            # Submit ALL tasks upfront — Parsl throttles execution via max_workers_per_node.
            logger.info(f"Submitting all {total_tasks} tasks to Parsl...")
            futures = []
            self._future_to_index = {}
            for idx, _ in indexed_partitions:
                f = self.submit_task(idx)
                futures.append(f)
                self._future_to_index[f] = idx
            logger.info(f"All {total_tasks} tasks submitted.")

            # Collect results as they complete.
            completed_count = 0
            processing_start = datetime.now()

            for job_index, result, duration, peak_memory_mb in self.wait_for_results(futures):
                completed_count += 1
                progress_pct = ((already_done + completed_count) / total_overall) * 100
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
                        "completed_count": already_done + completed_count,
                        "total_tasks": total_overall,
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
                    f"[{already_done + completed_count}/{total_overall}] ({progress_pct:.1f}%) "
                    f"task {job_index} done in {duration:.0f}s "
                    f"— elapsed {elapsed_str} — ETA {eta_str}"
                )
                logger.info(msg)
                print(msg, flush=True)
                if job_index >= 0:
                    self._mark_completed(job_index)
                results.append(result)

        finally:
            # Restore original signal handlers unconditionally.
            signal.signal(signal.SIGINT, _orig_sigint)
            signal.signal(signal.SIGTERM, _orig_sigterm)
            if _interrupted[0]:
                logger.warning("Interrupted — shutting down Parsl workers, please wait...")
            self.cleanup()
            if not _interrupted[0]:
                self._write_execution_summary(results)
                logger.info(f"Parallel execution completed with {self.name}")

        return results

    def cleanup(self) -> None:
        """Cleanup Parsl resources."""
        try:
            parsl.dfk().cleanup()
            parsl.clear()
            logger.info("Parsl HighThroughputExecutor cleaned up")
        except Exception as e:
            logger.warning(f"Error during Parsl cleanup: {e}")
