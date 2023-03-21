FROM python:latest
LABEL org.opencontainers.image.authors="Marc Grunberg"
LABEL version="0.1"

COPY requirements.txt requirements.txt
COPY patch/ patch/

RUN apt update && \
    apt upgrade -y && \
    apt-get install -y --no-install-recommends cmake && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt 

ARG NLL_PATH=/usr/local/nll
RUN svn export https://github.com/alomax/NonLinLoc/trunk/src ${NLL_PATH} && \
    patch ${NLL_PATH}/NLLocLib.c < patch/nll.patch && \
    cd ${NLL_PATH} && \
    cmake . && \
    make

ENV PATH="$PATH:${NLL_PATH}/bin"


