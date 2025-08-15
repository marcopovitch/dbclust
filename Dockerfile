FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends\
    build-essential \
    gfortran \
    curl \
    git \
    libspatialite7 \
    libsqlite3-mod-spatialite \
    pkg-config \
    libgomp1 \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Add uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# uv workdir
WORKDIR /app

# Copy project files
COPY . /app

# NonLinLoc
RUN git clone --depth=1 https://github.com/ut-beg-texnet/NonLinLoc /opt/nll && \
    cd /opt/nll/src && \
    rm -rf bin && \
    mkdir bin && \
    rm -f CMakeCache.txt

# Apply patch and build NonLinLoc
# fix arm run issue
RUN cd /opt/nll/src && \
    patch -p0 < /app/patch/nll/NLLocLib.patch && \
    cmake . && \
    make && \
    ln -s /opt/nll/src/bin /opt/nll/bin

ENV PATH="/opt/nll/bin:${PATH}"

# DBClust installation
RUN /bin/uv pip install --system --editable .

# patch Obspy/nlloc
# fix multiple picks from the same station in obs file
RUN patch --batch --forward \
    $(python3 -c "import obspy.io.nlloc.core; print(obspy.io.nlloc.core.__file__)") < /app/patch/obspy/obspy.io.nlloc.core.py.patch

# Ray.io dashboard and FastAPI port for fdsnws
EXPOSE 8000 8265

CMD ["dbclust"]
