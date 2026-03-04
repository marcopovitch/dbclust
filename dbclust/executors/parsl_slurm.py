"""Parsl SLURM executor implementation for DBClust.

This executor uses Parsl with HighThroughputExecutor and SlurmProvider
for execution on SLURM-managed HPC clusters.
"""

import logging
from typing import Any

from parsl.addresses import address_by_hostname
from parsl.launchers import SrunLauncher
from parsl.providers import SlurmProvider

from dbclust.config import DBClustConfig
from dbclust.executors.parsl_hte import ParslHTEExecutor

logger = logging.getLogger("dbclust")


class ParslSlurmExecutor(ParslHTEExecutor):
    """Parsl executor using HighThroughputExecutor with SlurmProvider.

    This executor extends ParslHTEExecutor to provide SLURM cluster support.
    It uses SlurmProvider with SrunLauncher for job submission on HPC clusters.

    Attributes:
        cfg: DBClust configuration object with slurm settings.
    """

    @property
    def name(self) -> str:
        return "Parsl SLURM (HighThroughputExecutor)"

    def _get_provider(self):
        """Get the SLURM provider for HTE.

        Returns:
            SlurmProvider configured from cfg.slurm settings.
        """
        slurm = self.cfg.slurm

        return SlurmProvider(
            partition=slurm.partition,
            account=slurm.account,
            nodes_per_block=slurm.nodes_per_block,
            cores_per_node=slurm.cores_per_node,
            min_blocks=slurm.min_blocks,
            max_blocks=slurm.max_blocks,
            walltime=slurm.walltime,
            worker_init=slurm.worker_init,
            scheduler_options=slurm.scheduler_options if slurm.scheduler_options else None,
            launcher=SrunLauncher(),
            cmd_timeout=120,
        )

    def initialize(self) -> None:
        """Initialize Parsl with HighThroughputExecutor and SlurmProvider."""
        import parsl
        from parsl.config import Config
        from parsl.executors import HighThroughputExecutor

        # Reduce Parsl logging noise
        if hasattr(parsl, "set_stream_logger"):
            parsl.set_stream_logger(level=logging.WARNING)

        for logger_name in [
            "parsl",
            "parsl.dataflow.dflow",
            "parsl.dataflow.memoization",
            "parsl.process_loggers",
            "parsl.jobs.strategy",
            "parsl.usage_tracking.usage",
        ]:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

        slurm = self.cfg.slurm

        # Configure HighThroughputExecutor with SLURM provider
        executor = HighThroughputExecutor(
            label="dbclust_slurm",
            address=address_by_hostname(),
            max_workers_per_node=slurm.max_workers_per_node,
            cores_per_worker=1,
            provider=self._get_provider(),
        )

        config = Config(
            executors=[executor],
            run_dir=self.cfg.parallel._temp_dir if self.cfg.parallel._temp_dir else "runinfo",
            retries=3,
            strategy="none",  # disable auto scale-in which causes ZMQError mid-run
        )

        parsl.load(config)
        logger.info(
            f"Parsl SLURM initialized: partition={slurm.partition}, "
            f"nodes_per_block={slurm.nodes_per_block}, "
            f"max_workers_per_node={slurm.max_workers_per_node}, "
            f"max_blocks={slurm.max_blocks}"
        )

    def cleanup(self) -> None:
        """Cleanup Parsl SLURM resources."""
        import parsl

        try:
            parsl.dfk().cleanup()
            parsl.clear()
            logger.info("Parsl SLURM executor cleaned up")
        except Exception as e:
            logger.warning(f"Error during Parsl SLURM cleanup: {e}")
