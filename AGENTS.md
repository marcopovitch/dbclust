# DBClust Agent Guide

This guide provides essential information for working with the DBClust seismic processing toolkit to avoid common pitfalls and accelerate onboarding.

## Core Commands

- `dbclust -c config.yml` - Main processing pipeline
- `nll-locate -c config.yml picks.csv output.qml` - Single event localization
- `injectdb -d events.db input.xml` - Import QuakeML into SQLite database
- `relocate -c config.yml input.qml output.qml` - Relocate existing events
- `fdsnws-server --db events.db` - Run FDSN web service
- `csv2parquet -i picks.csv -o picks.parquet` - Convert CSV to Parquet

## Project Structure

- Main modules: `dbclust/`, `fdsnws/`, `Utils/`
- Configuration: YAML files (e.g., `config.yml`)
- Data formats: Parquet, CSV, QuakeML, SQLite
- Parallel execution backends: Dask, Ray, Parsl

## Key Requirements

### NonLinLoc Installation
DBClust requires NonLinLoc binaries (NLLoc and scat2latlon) which must be built manually or via the provided script:
```bash
./nll_install.sh  # Builds NonLinLoc in $HOME/github/nll/bin
```

### Environment Setup
- Python 3.11+ required (3.12 recommended)
- Use `uv` for installation: `uv pip install -e .`
- For macOS: Set environment variables to avoid multiprocessing crashes:
  ```bash
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export VECLIB_MAXIMUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  ```

## Testing

Single test execution:
```bash
python test.py  # Run basic multiprocessing test
```

## Parallel Execution

### Executor Types
- `dask` - Best performance on local machines
- `ray` - Good performance with dashboard
- `parsl_hte` - Multi-process parallelism
- `parsl_thread` - Limited by Python GIL
- `parsl_slurm` - Experimental HPC support

### macOS Considerations
Due to platform limitations, macOS users should:
1. Run inside Docker for reliable parallel execution
2. Or use environment variables to limit internal threading
3. Avoid multiprocessing-based runtimes (Ray, Dask, Parsl HTE) with multithreaded numerical libraries

## Docker Usage

Build and run with Docker:
```bash
docker build -t dbclust:latest .
docker run -it --rm -v $(pwd):/app dbclust dbclust -c config.yml
```

Or use Docker Compose:
```bash
docker-compose up -d dbclust
```

## Important Notes

- DBClust uses a small SQLite database for intermediate storage
- The main processing pipeline follows: Data Loading → Pick Preprocessing → Station Processing → Time Window Processing → Clustering → First Pass Localization → Pick Filtering → Pick Relabeling → Second Pass Location → Output Generation
- Configuration is managed through YAML files with extensive options for parallel processing
- The system requires specific patches for NonLinLoc and ObsPy to work correctly with the processing pipeline