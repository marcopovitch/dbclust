#!/bin/bash
#set -x

# required
# - eventfetcher: https://github.com/marcopovitch/event-fetcher
# - dataselect:   https://github.com/earthscope/dataselect
# - msi:          https://github.com/EarthScope/msi
# - phasenet:     https://github.com/AI4EPS/PhaseNet


# Get mseed data
#./get_mseed_archive.py -e fr2022jyttas -l 180

PHASENET_DIR="${HOME}/github/PhaseNet"
DATA_DIR="Data"
MSEED_DIR="${DATA_DIR}/mseed"

mkdir -p ${DATA_DIR}/picks || rm -f ${DATA_DIR}/picks/*

# Write mseed file to be processed by phaseNet
# 3 channels (only) must be in the same file
#rm -f ${DATA_DIR}/mseed/*
#MSEED_FILE="${HOME}/github/event-fetcher/fr2022jyttas.mseed"
#dataselect -A ${MSEED_DIR}/%n.%s.mseed $MSEED_FILE

# check if each file has exactly 3 channels
error=0
for i in $(ls -1 ${MSEED_DIR}/*.mseed); do 
        nb=$(msi -T $i | wc -l); 
        if [ "$nb" -ne "5" ]; then 
                echo "Error($nb) with $i"; 
                let error++
        fi; 
done

if [ "$error" -ne "0" ]; then
        echo "Remove files above from processing."
        echo "phaseNet could have problems dealing with them !"
        exit
fi


# generate  network, station location, channel file
# to be used to convert NonLinLoc localisation to full QuakeML
# '_' is the separaror !!!
msi -T $MSEED_DIR/*  | \
        tail -n +2 | head -n-1 | cut -f1 -d' ' | \
        sort -u \
> ${DATA_DIR}/chan.txt

# Combine all EQT picks in one big file
# find . -name "*.csv" -exec cat {} >> EQT-2022-09-10.csv \;

# generate  network, station location, channel file from eqt file
# '_' is the separaror !!!
#cut -f1 -d, EQT-2022-09-10.csv | cut -f2 -d/ | cut -f1 -d_ | sort -u | tr "." "_" | grep -v file > ${DATA_DIR}/chan.txt


# generate csv (mseed files list) file for phaseNet
mkdir -p ${DATA_DIR}/csv
echo fname > ${DATA_DIR}/csv/mseed.csv
ls -1 ${MSEED_DIR} | grep -v "XX.GP" >> ${DATA_DIR}/csv/mseed.csv

mkdir -p ${DATA_DIR}/picks
python ${PHASENET_DIR}/phasenet/predict.py \
        --batch_size=1 \
        --model=${PHASENET_DIR}/model/190703-214543 \
        --data_dir=${DATA_DIR}/mseed  \
        --data_list=${DATA_DIR}/csv/mseed.csv \
        --format=mseed \
        --result_dir=${DATA_DIR}/picks \
        --result_fname=picks \
        --highpass_filter=4


 #rm -rf  Data/obs Data/tmp Data/qml; 
 #./dbclust.py -c dbclust.yml


# # generate csv file
# mkdir -p ${DATA_DIR}/csv
# for i in $(ls -1 ${MSEED_DIR}); do
#         echo fname > ${DATA_DIR}/csv/$i.csv;
#         echo $i >> ${DATA_DIR}/csv/$i.csv ;
# done

# # launch phaseNet on each individual stations
# mkdir -p ${DATA_DIR}/picks
# for i in $(ls -1 ${DATA_DIR}/csv); do
#         echo $i;
#         python ${PHASENET_DIR}/phasenet/predict.py --model=${PHASENET_DIR}/model/190703-214543 \
#                 --data_dir=${DATA_DIR}/mseed  \
#                 --data_list=${DATA_DIR}/csv/$i \
#                 --format=mseed \
#                 --result_dir=${DATA_DIR}/picks \
#                 --result_fname=$i.picks \
#                 --highpass_filter=4

# # Header
# echo "file_name,begin_time,station_id,phase_index,phase_time,phase_score,phase_type" > ${DATA_DIR}/picks/picks.csv
# # concatenate results
# (for i in $(ls -1 ${DATA_DIR}/picks/*.csv); do 
# 	tail -n +2  $i; 
# done) >> ${DATA_DIR}/picks.csv
