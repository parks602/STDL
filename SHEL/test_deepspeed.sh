#!/bin/sh
CDIR="/home/ubuntu/pkw/PKW_STDL"
SRCS=${CDIR}/SRCS
SHEL=${CDIR}/SHEL
DABA=${CDIR}/DABA
DAOU=${CDIR}/DAOU
DAIN=${CDIR}/DAIN
DDIR=${CDIR}/DATA
ODIR=${DAOU}/LDAPS_OBS_200000_stratify/T3H
MODL=${ODIR}/checkpoint.pt

DATA=${DDIR}/2019010100-2020010100_T3H_200000_stratify_data.csv
INFO=${DDIR}/2019010100-2020010100_T3H_200000_stratify_info.csv

MESH=${DDIR}/mesh_1km.csv

python  ${SRCS}/test_deepspeed.py --outf ${ODIR} --net ${MODL} \
	                          --dataf ${DATA} --infof ${INFO} \
				  --meshf ${MESH}
