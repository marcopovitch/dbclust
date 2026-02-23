FROM python:3.12-slim-bookworm

LABEL maintainer="Marc Grunberg <marc.grunberg@unistra.fr>" version="0.1"

# System dependencies 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    curl \
    git \
    libspatialite7 \
    libsqlite3-mod-spatialite \
    pkg-config \
    libgomp1 \
    cmake \
    vim \
    visidata \
    && rm -rf /var/lib/apt/lists/*

# Add uv 
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# NonLinLoc build 
RUN git clone --depth=1 https://github.com/ut-beg-texnet/NonLinLoc /opt/nll && \
    cd /opt/nll/src && \
    rm -rf bin && \
    mkdir bin && \
    rm -f CMakeCache.txt

# Copy only the patch files needed for NLL build
COPY patch/nll/NLLocLib.patch /tmp/NLLocLib.patch

# Apply patch and build NonLinLoc
RUN cd /opt/nll/src && \
    patch -p0 < /tmp/NLLocLib.patch && \
    cmake . && \
    make && \
    ln -s /opt/nll/src/bin /opt/nll/bin && \
    rm /tmp/NLLocLib.patch

ENV PATH="/opt/nll/bin:${PATH}"

# Set workdir
WORKDIR /app

# Copy dependency files first 
COPY pyproject.toml uv.lock* ./

# Install project dependencies.
# pyrocko doesn't declare pkg_resources as a build dependency; this is handled
# via [tool.uv.extra-build-dependencies] in pyproject.toml.
RUN /bin/uv pip install --system .

# Copy patch files for obspy
COPY patch/obspy/obspy.io.nlloc.core.py.patch /tmp/obspy.patch

# Apply obspy patch (use find instead of python import to locate the file)
RUN patch --batch --forward \
    $(find /usr/local/lib -path '*/obspy/io/nlloc/core.py' -print -quit) < /tmp/obspy.patch && \
    rm /tmp/obspy.patch

# Copy application code last 
COPY dbclust/ ./dbclust/

# Ensure dbclust module is in Python path for Parsl HTE workers
ENV PYTHONPATH="/app"

# Ray.io dashboard and FastAPI port for fdsnws
EXPOSE 8265

CMD ["dbclust"]
