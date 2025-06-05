# DBClust - Seismic Phase Association and Event Localization

DBClust is a powerful tool for seismic phase association and event localization. It is designed to process huge pick datasets (parquet, csv) and automatically identify seismic events using advanced clustering algorithms. It integrates with NonLinLoc for earthquake localization and provides a flexible framework for seismic data analysis through a small SQlite database and a FDSN web service.

## Features

- **Advanced Clustering**: Utilizes HDBSCAN and PyOcto (OPTICS, DBSCAN can be used as well) algorithm for robust phase association.
- **Seismic Event Localization**: Uses NonLinLoc for precise earthquake localization
- **Flexible Configuration**: Highly configurable through YAML configuration files
- **Parallel Processing**: Supports parallel processing for performance
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
uv pip install -e ".[dev]"
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

and use your browser to access to ray.io dashboard `http://localhost:8265`.

For the fdsnws service, use:

```bash
docker-compose up -d fdsnws
```

and use your browser to access to the fdsnws service at `http://localhost:8000`.


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

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.

## Acknowledgements

- [NonLinLoc](http://alomax.free.fr/nlloc/) - Earthquake location algorithm
- [PyOcto](https://github.com/yetinam/pyocto) - PyOcto, a seismic phase associator
- [ObsPy](https://github.com/obspy/obspy) - Python framework for seismology
- [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) - Hierarchical DBSCAN clustering
- [scikit-learn](https://scikit-learn.org/) - Machine learning in Python
