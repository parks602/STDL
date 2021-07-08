#!/bin/sh

CDIR="/home/onlyred/KMA/ST_DL"
SRCS=${CDIR}/SRCS
SHEL=${CDIR}/SHEL
DABA=${CDIR}/DABA
DAOU=${CDIR}/DAOU
DAIN=${CDIR}/DAIN

CONF=${DABA}/deepspeed_config.json
ODIR=${DAOU}/BUOY_TEST_1km

DATA=${DAIN}/aws+buoy_spatio_temporal_temperature.csv
INFO=${DAIN}/aws+buoy_stninfo.csv
MESH=${DAIN}/mesh_1km.csv

python  ${SRCS}/Experiment_torch.py --cuda --epochs 2000 --lr 0.001 --batchSize 32 \
			    --outf ${ODIR} --dataf ${DATA} --infof ${INFO} --meshf ${MESH}
