# DBClust - Seismic Phase Association and Event Localization

DBClust is a powerful tool for seismic phase association and event localization. It is designed to process huge pick datasets (parquet, csv) and automatically identify seismic events using advanced clustering algorithms. It integrates with NonLinLoc for earthquake localization and provides a flexible framework for seismic data analysis through a small SQlite database and a FDSN web service.

## Features

- **Advanced Clustering**: Utilizes HDBSCAN, Leiden and PyOcto (OPTICS, DBSCAN can be used as well) algorithms for robust phase association.
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

### NonLinLoc Installation

DBClust requires NonLinLoc for earthquake localization. You can install it using the provided script:

```bash
# Clone and build NonLinLoc
./nll_install.sh
```

The script will:
- Clone the NonLinLoc repository
- Apply necessary patches
- Build the binaries
- Create symlinks to `$HOME/github/nll/bin`

**Manual Installation** (if the script doesn't work):
```bash
# Clone NonLinLoc
git clone --depth=1 https://github.com/ut-beg-texnet/NonLinLoc $HOME/github/nll
cd $HOME/github/nll/src

# Apply patch (if available)
cp patch/nll/NLLocLib.patch .
patch -p0 < NLLocLib.patch

# Build
cmake .
make
ln -s $HOME/github/nll/src/bin $HOME/github/nll/bin
```

After installation, ensure the NonLinLoc binaries are in your PATH:
```bash
export PATH=$HOME/github/nll/bin:$PATH
```

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

### Command Line Tools

DBClust provides several command-line tools for different seismic data processing tasks:

### Core Tools

| Tool | Description | Command |
|------|-------------|---------|
| **dbclust** | Main seismic event detection and localization pipeline | `dbclust -c config.yml` |
| **nll-locate** | Single event localization using NonLinLoc | `nll-locate -c config.yml picks.csv output.qml` |
| **injectdb** | Import QuakeML files into SQLite database | `injectdb -d events.db input.xml` |
| **relocate** | Relocate events from QuakeML files | `relocate -c config.yml input.qml output.qml` |

### Utility Tools

| Tool | Description | Command |
|------|-------------|---------|
| **fdsnws-server** | FDSN Web Service server for event access | `fdsnws-server -d events.db -p 8000` |
| **csv2parquet** | Convert CSV files to Parquet format | `csv2parquet -i input.csv -o output.parquet` |
| **add-event-types** | Enrich the catalog with per-agency event_type columns and a consensus | `add-event-types -c add_event_types.yml` |
| **pick_stats_from_config** | Pick statistics (manual/auto counts, stations) from a YAML config | `python Utils/pick_stats_from_config.py -c config.yml` |

### Utility Scripts

Scripts run directly with `python Utils/...` (not installed as CLI entry points):

| Script                           | Description                                                                | Command                                                |
| -------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------ |
| **detect_operator_duplicates**   | Detect events picked independently by two operators on the same earthquake | `python Utils/detect_operator_duplicates.py events.db` |
| **detect_suspicious_duplicates** | Detect intra-window Leiden fragment duplicates (auto/manual pick conflict) | Called automatically during `merge_databases()`        |

### dbclust - Main Processing Pipeline

The main tool processes seismic picks to detect and locate events:

```bash
# Basic usage
dbclust -c config.yml

# With specific velocity profile and log level
dbclust -c config.yml -p "local_model" -l DEBUG

# Example configuration file structure
cat config.yml
catalog:
  sqlite_db_fullpath: "seismic_events.db"
  picks_file: "picks.parquet"
  keep_temp_db: false

parallel:
  n_workers: 12
  partition_duration: "1D"
  executor: "dask"

localization:
  nll_template: "nll_template.in"
  time_grid: "times.grid"
```

### nll-locate - Single Event Localization

Localize individual events using NonLinLoc:

```bash
# Basic localization
nll-locate -c config.yml picks.csv output.qml

# With custom uncertainties
nll-locate -c config.yml -p 0.1 -s 0.2 picks.csv output.qml

# Using specific NonLinLoc template
nll-locate -c config.yml -t custom_template.in picks.csv output.qml
```

### injectdb - QuakeML Database Import

Import QuakeML files into a SpatiaLite-enabled SQLite database:

```bash
# Import single file
injectdb -d events.db input.xml

# Import multiple files
injectdb -d events.db input1.xml input2.xml input3.xml

# Import from file list (for large numbers of files)
injectdb -d events.db --input-list filelist.txt

# Fast import mode (for initial bulk loading)
injectdb -d events.db --input-list filelist.txt --fast-import

# With batching for large datasets
injectdb -d events.db --input-list filelist.txt --batch-size 10000 --sqlite-batch-size 10000

# Store full QuakeML data
injectdb -d events.db input.xml -q

# Enable database enhancements
injectdb -d events.db input.xml --compute-ps-ratio --compute-station_scores
```

#### Database Enhancement Options

The `injectdb` tool supports various database enhancement options:

```bash
# Compute station scores and ps_ratio
injectdb -d events.db --compute-station-scores --compute-ps-ratio

# Add spectrocnn discrimination info from predictions CSV
injectdb -d events.db --add-discrimination event-predictions.csv

# Add full multi-agency event_type enrichment from enriched alceste CSV
injectdb -d events.db --add-event-type-enrichment alceste-with_event_types.csv

# Compute localization quality metrics (also populates nll_epicenters_diff_km)
injectdb -d events.db --add-localization-quality

# Import silence scores from a silence-score CSV
injectdb -d events.db --add-silence-score silence.csv

# Add agency names
injectdb -d events.db --add-agency-names

# Compute GT5 metrics
injectdb -d events.db --gt5

# Compute median probabilities
injectdb -d events.db --compute-prob-median

# Refresh views
injectdb -d events.db --refresh-view
```

#### Recommended order for full event_type enrichment

Both enhancement steps are required and **neither is optional or obsolete**:
`--add-discrimination` is the only step that populates the quantitative
spectrocnn columns (`spectrocnn_probability`, `spectrocnn_station_count`,
`spectrocnn_certainty`). These are essential to assess discrimination quality
for PHASENET-only events (no contributing agency bulletin), since for those
events spectrocnn is the only source of an `event_type` classification.

```bash
# 1. Generate enriched CSV (spectrocnn + agency bulletins)
add-event-types -c add_event_types.yml

# 2. Multi-source final classification (event_type, event_type_source,
#    event_type_consensus, event_type_agencies_json) — does not touch the
#    spectrocnn_* columns.
injectdb -d events.db --add-event-type-enrichment alceste-with_event_types.csv

# 3. Spectrocnn quantitative metrics (spectrocnn_probability,
#    spectrocnn_station_count, spectrocnn_certainty) — always written,
#    regardless of whether event_type was already set in step 2.
injectdb -d events.db --add-discrimination event-predictions.csv
```

The two `injectdb` calls can be run in either order: each one only ever writes
its own set of columns and never re-reads the other's output.

- `--add-discrimination` always populates `spectrocnn_probability`,
  `spectrocnn_station_count`, and `spectrocnn_certainty` from the spectrocnn
  predictions CSV, for every event present in that CSV — whether or not
  `event_type` is already set. It additionally sets `event_type` (and
  `event_type_source = 'spectrocnn'`, `event_type_agencies_json`) but **only**
  when `event_type` is still NULL, so it never overwrites a value already set
  by `--add-event-type-enrichment`.

- `--add-event-type-enrichment` always overwrites `event_type`,
  `event_type_source`, `event_type_consensus`, and `event_type_agencies_json`.
  It requires the enriched alceste CSV produced by `add-event-types` and never
  touches the `spectrocnn_*` columns.

#### silence score (`--add-silence-score`)

Imports pre-computed silence scores from the output CSV of the
[silence-score](https://gitlab.com/marcopovitch/silence_score) tool.
The silence score measures the fraction of active nearby stations that did not
contribute to the event detection (0 = all active stations contributed,
1 = none did). High values indicate suspicious or poorly located events.

The CSV must contain at least: `event_id`, `score`, `n_candidates`, `n_excluded`,
`n_used`, `n_used_outside_radius`, `n_active`, `n_active_missing`,
`expected_weight`, `missing_weight`, `effective_radius_km`, `reason`,
`radius_km`, `threshold`, `decay_factor`.

When `origin_id` is absent from the CSV, the preferred origin of each event is
used. Results are stored in a dedicated `silence_scores` table (keyed on
`origin_id`) and exposed as `silence_score` in `event_coordinates`.

#### NLL epicenters offset (`nll_epicenters_diff_km`)

Populated automatically during `--add-localization-quality` (and at injection
time for new imports). Stores the horizontal distance (km) between the two
NonLinLoc localizations produced for each origin:

- **scatter centroid** — location derived from the scatter cloud ellipse
- **PDF maximum** — location at the maximum of the probability density function

A large offset indicates an asymmetric or multimodal location PDF. The column
is stored in the `origins` table and exposed in `event_coordinates` just after
`scatter_volume`.

#### Export Options

```bash
# Export to CSV
injectdb -d events.db -c events.csv

# Export to QuakeML
injectdb -d events.db --export-quakeml events.qml

# Export specific events
injectdb -d events.db --export-quakeml events.qml -e event1 event2 event3

# Export with time range
injectdb -d events.db --export-quakeml events.qml --start-time 2023-01-01 --end-time 2023-12-31

# Export monthly files
injectdb -d events.db --export-quakeml output_dir/
```

### relocate - Event Relocation

Relocate events from existing QuakeML files:

```bash
# Basic relocation
relocate -c config.yml input.qml output.qml

# With specific velocity profile
relocate -c config.yml -p "custom_model" input.qml output.qml
```

### fdsnws-server - FDSN Web Service

Run an FDSN event web service:

```bash
# Basic server
fdsnws-server -d events.db

# Custom port
fdsnws-server -d events.db -p 8080

# With custom host
fdsnws-server -d events.db -p 8000 --host 0.0.0.0
```

Access the service at:
- Events: `http://localhost:8000/fdsnws/event/1/query`
- Catalog: `http://localhost:8000/fdsnws/event/1/catalog`
- Built-in browser: `http://localhost:8000`

### csv2parquet - File Format Conversion

Convert CSV pick files to Parquet format for better performance:

```bash
# Basic conversion
csv2parquet -i picks.csv -o picks.parquet

# With compression
csv2parquet -i picks.csv -o picks.parquet --compression snappy
```

### add-event-types - Multi-source Event Type Enrichment

Produce an enriched catalog CSV with per-agency and spectrocnn `event_type` columns,
a consensus, and a final canonical value with full provenance tracking.

```bash
add-event-types -c add_event_types.yml
```

**Per-agency columns** (`event_type_<AGENCY>`) are derived from each agency's bulletin
CSV (joined via `agencies_list`/`agency_names`) or from a fixed value for agencies that
only contribute one event type (e.g. `earthquake` for ISTERRE).

**Spectrocnn column** (`event_type_SPECTROCNN`) is populated directly from the raw
predictions file (`spectrocnn_predictions_file` in the config: `predhdq50` 0=earthquake,
1=quarry blast). This avoids contamination from a previously enriched `event_type` column
in the catalog CSV.

**Output CSV columns** (produced by `add-event-types`, consumed by `--add-event-type-enrichment`):

| CSV column               | DB column (after enrichment) | Description                                                                                                     |
| ------------------------ | ---------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `event_type_<AGENCY>`    | `event_type_agencies` table  | Per-agency classification                                                                                       |
| `event_type_SPECTROCNN`  | `event_type_agencies` table  | Spectrocnn classification (earthquake / quarry blast)                                                           |
| `event_type_consensus`   | `event_type_consensus`       | Inter-agency agreement: `consensus` / `conflict` / `no_data` / `NULL` (NULL = spectrocnn-only, no agency data) |
| `event_type_final`       | `event_type`                 | Canonical QuakeML event type — final value combining all sources                                                |
| `event_type_final_source` | `event_type_source`          | Provenance of the final value (see table below)                                                                 |

> Note: `event_type_final` in the CSV becomes `event_type` in the database (QuakeML field name).
> Similarly `event_type_final_source` → `event_type_source`.

**`event_type_source` provenance values:**

| Value | Meaning |
| ----- | ------- |
| `spectrocnn` | spectrocnn only, no compatible agency refinement |
| `spectrocnn+consensus` | spectrocnn category, refined by agency consensus |
| `spectrocnn+single:<AGENCY>` | spectrocnn category, refined by single agency |
| `spectrocnn+conflict:<AGENCY>` | spectrocnn category, refined after conflict resolution |
| `consensus` | multiple agencies agreed, no spectrocnn |
| `single:<AGENCY>` | only one agency had a value, no spectrocnn |
| `conflict:<AGENCY>` | agencies disagreed, `<AGENCY>` resolved it |

**Priority rules:**

- spectrocnn always takes priority over agency values
- when the agency value is a compatible refinement of the spectrocnn category (e.g.
  `induced or triggered event` is a refinement of `earthquake`), the more precise
  agency label is used
- inter-agency conflicts are resolved by `agency_priority` (config key: ordered list,
  e.g. `[RENASS, LDG]`, then config order)

The config file (`add_event_types.yml`) specifies `input_file`, `output_file`,
`spectrocnn_predictions_file`, `agency_priority`, `spectrocnn_compatibility`,
`event_type_groups` (normalization map), and the `agencies` list.
See `Utils/add_event_types.yml` for a full example.

### pick_stats_from_config - Pick Statistics

Compute pick statistics from the parquet pick files declared in a DBClust YAML config,
applying the same P/S probability thresholds, station blacklist, and rename rules as the pipeline.

```bash
python Utils/pick_stats_from_config.py -c config.yml \
    --stations-csv stations.csv \
    --missing-stations-csv missing.csv \
    --suggest-rename-yaml suggested_renames.yml
```

Options:

| Option | Description |
|--------|-------------|
| `-c` | Path to DBClust YAML config (required) |
| `--stations-csv FILE` | Export stations with coordinates, `manual_picks`, and `auto_picks` columns |
| `--missing-stations-csv FILE` | Export stations found in picks but absent from inventory/fallback, with `pick_count`, `agencies`, and `suggested_rename` columns |
| `--suggest-rename-yaml FILE` | Write suggested `!extend`-ready rename rules for missing stations that exist under a different network code |

The suggested rename YAML can be included directly in the config via `!extend`:

```yaml
station:
  rename:
    before:
      - ^(FR\..*?\..*?)\.XXZ: \1.SH
      - !extend suggested_renames.yml   # merged inline
```

### detect_operator_duplicates - Operator Duplicate Detection

Detect events that were independently picked by two operators on the same earthquake.
These events have all-manual picks on the same stations/phases but with slightly
different timestamps (typically < 1s). They require manual review and correction.

```bash
# Basic usage — scans full database, writes JSON report
python Utils/detect_operator_duplicates.py events.db

# Custom thresholds
python Utils/detect_operator_duplicates.py events.db \
    --max-origin-dt 5.0 \   # max time between origins (s)
    --max-dist-km 15.0 \    # max distance between hypocentres (km)
    --max-pick-dt 1.0 \     # max time diff between matching picks (s)
    --min-stations 3        # min shared stations to flag

# With CSV output for spreadsheet review
python Utils/detect_operator_duplicates.py events.db \
    --output duplicates.json \
    --csv duplicates.csv
```

Output JSON structure:

```json
{
  "summary": {"n_duplicates": 17, "db": "...", "generated_at": "..."},
  "duplicates": [
    {
      "event_id_1": "...", "origin_time_1": "...", "phases_1": 33,
      "event_id_2": "...", "origin_time_2": "...", "phases_2": 14,
      "dt_s": 0.163, "dist_km": 6.5,
      "n_shared_stations": 9, "max_pick_dt_s": 0.612,
      "note": "Operator duplicate suspected — manual review recommended"
    }
  ]
}
```

Detection uses an O(n log n) sliding-window scan on the origins table sorted by time,
so performance scales well even on large catalogues (15 years / 150K events in ~15s).

### detect_suspicious_duplicates - Intra-window Fragment Detection

Detects same-window Leiden fragment duplicates caused by co-existing automatic DL picks
and manual picks on the same stations (picks differ by 0.1–0.5s, just above the
deduplication threshold). Unlike operator duplicates these cannot be safely removed
automatically — a WARNING is logged and they appear in the merge report.

This function is called automatically at the end of `merge_databases()` and its results
are included in the `.dedup_report.json` report written alongside the final database.

```json
{
  "suspicious_duplicates": [
    {
      "event_id_1": "...", "phases_1": 8,
      "event_id_2": "...", "phases_2": 28,
      "n_conflict_stations": 5, "dist_km": 4.5, "dt_s": 0.66,
      "note": "Auto/manual pick conflict on shared stations — manual review recommended"
    }
  ]
}
```

## Performance Optimization

### Large Dataset Import Performance

When importing large numbers of QuakeML files, use these optimization strategies:

#### Fast Import Mode

For initial bulk loading of reliable data:
```bash
injectdb -d events.db --input-list filelist.txt --fast-import
```

Fast import mode:
- Disables foreign key constraints during import
- Uses aggressive SQLite pragmas (MEMORY journal, OFF synchronous)
- Significantly improves import speed for large datasets
- **Warning**: Reduces durability, use only for initial bulk loading

#### Batching Strategies

For optimal performance with large datasets:
```bash
# Large batch sizes for better throughput
injectdb -d events.db --input-list filelist.txt \
  --batch-size 10000 \
  --sqlite-batch-size 10000 \
  --fast-import
```

- `--batch-size`: Events accumulated in memory before database insertion
- `--sqlite-batch-size`: Events per SQLite transaction commit
- Larger batches reduce transaction overhead but use more memory

#### Database Enhancements

Compute performance metrics after import:
```bash
# Compute station scores and ps_ratio for better analysis
injectdb -d events.db --compute-station-scores --compute-ps-ratio

# Add agency names for better event categorization
injectdb -d events.db --add-agency-names
```

### Parallel Processing Configuration

Configure parallel execution based on your system:

```yaml
# For local machines (best performance)
parallel:
  n_workers: 12  # Number of CPU cores
  partition_duration: "1D"
  executor: "dask"

# For HPC clusters
parallel:
  n_workers: 32
  partition_duration: "6H"
  executor: "parsl_slurm"

slurm:
  enabled: true
  partition: "compute"
  cores_per_node: 32
  walltime: "72:00:00"
```

### Memory Optimization

For memory-constrained systems:
```yaml
# Reduce memory usage
parallel:
  n_workers: 4  # Fewer workers
  partition_duration: "6H"  # Shorter windows
  
catalog:
  keep_temp_db: false  # Clean up temporary databases
```

## Processing Pipeline

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
