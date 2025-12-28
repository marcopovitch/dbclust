"""DBClust executors package.

This package provides different parallel execution backends for DBClust:

- ParslThreadExecutor: Uses Parsl with ThreadPoolExecutor (default)
- ParslHTEExecutor: Uses Parsl with HighThroughputExecutor
- ParslSlurmExecutor: Uses Parsl with HighThroughputExecutor + SlurmProvider
- RayExecutor: Uses Ray distributed computing
- DaskExecutor: Uses Dask LocalCluster

Usage:
    from dbclust.executors import get_executor

    executor = get_executor(cfg)
    results = executor.run()
"""

from dbclust.executors.base import ExecutorBase

__all__ = ["ExecutorBase", "get_executor"]


def get_executor(cfg):
    """Factory function to select executor based on configuration.

    Args:
        cfg: DBClustConfig instance containing parallel and slurm settings.

    Returns:
        An executor instance appropriate for the configuration.

    Raises:
        ValueError: If the specified executor type is unknown.
    """
    # Check for SLURM first (takes precedence)
    if hasattr(cfg, 'slurm') and cfg.slurm and cfg.slurm.enabled:
        from dbclust.executors.parsl_slurm import ParslSlurmExecutor
        return ParslSlurmExecutor(cfg)

    # Get executor type from config, default to parsl_thread
    executor_type = getattr(cfg.parallel, 'executor', 'parsl_thread')

    if executor_type == 'ray':
        from dbclust.executors.ray_executor import RayExecutor
        return RayExecutor(cfg)
    elif executor_type == 'parsl_hte':
        from dbclust.executors.parsl_hte import ParslHTEExecutor
        return ParslHTEExecutor(cfg)
    elif executor_type == 'parsl_thread':
        from dbclust.executors.parsl_thread import ParslThreadExecutor
        return ParslThreadExecutor(cfg)
    elif executor_type == 'dask':
        from dbclust.executors.dask_executor import DaskExecutor
        return DaskExecutor(cfg)
    else:
        raise ValueError(f"Unknown executor type: {executor_type}. "
                         f"Valid options: ray, parsl_hte, parsl_thread, dask")
