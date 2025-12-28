# DBClust - Seismic Phase Association and Event Localization

DBClust is a powerful tool for seismic phase association and event localization. It is designed to process huge pick datasets (parquet, csv) and automatically identify seismic events using advanced clustering algorithms. It integrates with NonLinLoc for earthquake localization and provides a flexible framework for seismic data analysis through a small SQlite database and a FDSN web service.

## Features

- **Advanced Clustering**: Utilizes HDBSCAN and PyOcto (OPTICS, DBSCAN can be used as well) algorithm for robust phase association.
- **Seismic Event Localization**: Uses NonLinLoc for precise earthquake localization
- **Flexible Configuration**: Highly configurable through YAML configuration files
- **Parallel Processing**: Supports multiple parallel execution backends (Dask, Ray, Parsl)
- **Multiple Output Formats**: Exports results in various formats including QuakeML and SQLite
- **Pick Preprocessing**: Processes seismic picks removing duplicates and filtering picks based on proximity threshold
- **Pick Filtering**: A posteriori filtering of seismic phase based on time residual threshold and more advanced criteria
- **Pick Relabeling**: Relabels seismic phase picks based on configurable pick zone and score threshold
- **Event Relocation**: Relocates seismic events from quakeml files using NonLinLoc
- **fdsnws event**: Provides a simple fdsnws event service to access seismic events. A builder and a browser are provided.
- **docker support**: A Dockerfile and docker-compose.yml are provided to run DBClust and fdsnws in a container.

## Prerequisites

- Python >= 3.10
- NonLinLoc binaries (NLLoc and scat2latlon)
- NonLinLoc configuration template file
- Time grid files for your region of interest
- Configuration file (yaml)

## Installation

### Using uv (recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver written in Rust.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install the package in development mode with all dependencies
uv pip install -e .
```

### Using pip

```bash
pip install -e .
```

### Using conda

```bash
conda create -n dbclust python=3.10
conda activate dbclust
pip install -e .
```

## Docker Setup

### Building the Docker Image

```bash
docker build -t dbclust:latest .
```

### Running the Container

To run DBClust in a Docker container, you'll need to mount:

1. Your local DBClust directory
2. NonLinLoc time grid files
3. Any additional data files needed

```bash
docker run -it --rm \
    -v $(pwd):/app \
    -v /path/to/nll/times:/nll_times \
    -v /path/to/your/data:/data \
    -w /app \
    dbclust \
    python -m dbclust --config /path/to/your/config.yml
```

### Docker Compose

Use and modify the `docker-compose.yml` file to run DBClust in a Docker container.

For the dbclust application, you can use:

```bash
docker-compose up -d dbclust
```

and use your browser to access the ray.io dashboard at `http://localhost:8265`.

For the fdsnws service, use:

```bash
docker-compose up -d fdsnws
```

and use your browser to access to the fdsnws service at `http://localhost:8000`.


## Updating Git Dependencies with uv

When using `dbclust` as a git dependency in another project (e.g., `dbclust @ git+https://github.com/marcopovitch/dbclust@dbclust2`), `uv sync` will **not** automatically fetch new commits. The `uv.lock` file pins dependencies to specific commit hashes for reproducibility.

To update to the latest version:

```bash
# Update only dbclust
uv lock --upgrade-package dbclust
uv sync

# Or update all packages
uv lock --upgrade
uv sync
```

If you encounter issues, you can reset everything:

```bash
# Nuclear option: delete lock, venv, and cache
rm -rf .venv uv.lock
uv cache clean
uv sync
```

## Usage

### Command-line Arguments

TBD

### Processing Pipeline

1. **Data Loading**: Load seismic phase picks from parquet, csv or obspy stream files
2. **Pick Preprocessing**: Remove duplicate picks and filter picks based on proximity threshold
3. **Station Processing**: Fetch station metadata and apply any renaming rules
4. **Time Window Processing**: Process data in configurable time windows
5. **Clustering**: Apply clustering algorithms to identify seismic events
6. **First Pass localization**: Use NonLinLoc to locate identified events
7. **Pick Filtering**: A posteriori filtering of seismic phase based on time residual threshold and more advanced criteria
8. **Pick Relabeling**: Relabel seismic phase picks based on configurable pick zone and score threshold
9. **Second Pass location**: Use NonLinLoc to relocate identified events
10. **Output**: Generate QuakeML files and update SQLite database

## Output

DBClust generates the following outputs:

- **QuakeML files**: Standardized earthquake catalog in QuakeML format
- **SQLite database**: Complete event catalog in a queryable SQLite database
- **Log files**: Detailed processing logs

## Parallel Execution

DBClust supports multiple parallel execution backends, configurable via the `parallel.executor` option in the YAML configuration file.

### Available Executors

| Executor | Description | Use Case |
|----------|-------------|----------|
| `dask` | Dask LocalCluster | Best performance on local machines |
| `ray` | Ray distributed computing | Good performance, includes dashboard |
| `parsl_hte` | Parsl HighThroughputExecutor | Multi-process parallelism |
| `parsl_thread` | Parsl ThreadPoolExecutor | Limited by Python GIL |
| `parsl_slurm` | Parsl with SLURM provider | HPC clusters (experimental) |

### Configuration Example

```yaml
parallel:
  n_workers: 12
  partition_duration: "1D"
  executor: "dask"  # dask, ray, parsl_hte, parsl_thread
```

### macOS Considerations

On macOS, using multiprocessing-based parallelism runtimes (Ray, Dask, Parsl HTE) in combination with multithreaded numerical libraries (NumPy/SciPy/ObsPy using OpenBLAS, MKL, or Accelerate) can cause native crashes (Bus error, Segmentation fault).

This issue is due to unsafe interactions between fork/spawn and BLAS libraries, which are not fully fork-safe on macOS. This is not a Python bug but a platform limitation.

**Running DBClust inside Docker is strongly recommended on macOS** for reliable parallel execution.

#### Recommended Setup for macOS

1. Use Docker (see [Docker Setup](#docker-setup))
2. Choose one of the following executors (ordered by performance):
   - `dask` - Best performance
   - `ray` - Good performance with monitoring dashboard
   - `parsl_hte` - Good multi-process parallelism

The `parsl_thread` executor is limited by Python's Global Interpreter Lock (GIL) and is not recommended for CPU-intensive workloads.

#### Environment Variables for Native macOS Execution

If you need to run outside Docker on macOS, these environment variables limit internal multithreading in numerical libraries to avoid unsafe interactions with multiprocessing on macOS, but they do not guarantee full stability in all cases.

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

These settings restrict internal threading in OpenMP, OpenBLAS, MKL, Apple’s vecLib, and NumExpr, helping to reduce thread contention and the risk of native crashes.

### SLURM Support (Experimental)

The `parsl_slurm` executor enables execution on HPC clusters with SLURM workload manager. This feature is **experimental and not yet fully tested**.

```yaml
parallel:
  n_workers: 32
  executor: "parsl_hte"  # Will use SLURM if slurm.enabled is true

slurm:
  enabled: true
  partition: "compute"
  account: "my_project"
  nodes_per_block: 1
  cores_per_node: 32
  max_workers_per_node: 32
  walltime: "72:00:00"
  worker_init: "module load python; conda activate dbclust"
  scheduler_options: ""
  max_blocks: 10
  min_blocks: 0
```

#### SLURM Configuration Options

| Option | Description |
|--------|-------------|
| `enabled` | Enable SLURM execution (`true`/`false`) |
| `partition` | SLURM partition name (e.g., `compute`, `gpu`) |
| `account` | SLURM account for billing (`-A` option), set to `null` if not required |
| `nodes_per_block` | Number of nodes per SLURM job block |
| `cores_per_node` | Number of CPU cores available per node |
| `max_workers_per_node` | Maximum parallel workers per node (typically equals `cores_per_node`) |
| `walltime` | Maximum job duration in `HH:MM:SS` format |
| `worker_init` | Shell commands to initialize the environment on compute nodes (e.g., load modules, activate conda) |
| `scheduler_options` | Additional SBATCH options (e.g., `#SBATCH --mem=128G\n#SBATCH --exclusive`) |
| `max_blocks` | Maximum number of concurrent SLURM job blocks (higher = more parallelism) |
| `min_blocks` | Minimum blocks to keep alive (`0` = scale down to zero when idle) |

## License

MIT License with Commons Clause.
Commercial use is prohibited without explicit permission from the author.

## Acknowledgements

### Seismology

- [NonLinLoc](http://alomax.free.fr/nlloc/) - Earthquake location algorithm
- [PyOcto](https://github.com/yetinam/pyocto) - Seismic phase associator
- [ObsPy](https://github.com/obspy/obspy) - Python framework for seismology

### Machine Learning

- [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) - Hierarchical DBSCAN clustering
- [scikit-learn](https://scikit-learn.org/) - Machine learning in Python

### Parallel Execution Backends

- [Dask](https://www.dask.org/) - Flexible parallel computing library
- [Ray](https://www.ray.io/) - Distributed computing framework
- [Parsl](https://parsl-project.org/) - Parallel scripting library for Python
