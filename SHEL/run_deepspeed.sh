#!/bin/sh

CDIR="/home/ubuntu/pkw/PKW_STDL"
LDAPSDIRHEAD="/home/ubuntu/pkw/DATA/KOR1KM"
VAR="T3H"
'''
VAR = T3H or REH
'''
LDAPSDIR=${LDAPSDIRHEAD}/${VAR}
SRCS=${CDIR}/SRCS
SHEL=${CDIR}/SHEL
DABA=${CDIR}/DABA
DAOU=${CDIR}/DAOU
DAIN=${CDIR}/DAIN
DDIR=${CDIR}/DATA
CONF=${DABA}/deepspeed_config.json
ODIR=${DAOU}/LDAPS_OBS_ALL/${VAR}
DATA=${DDIR}/${VAR}_all_data.csv
INFO=${DDIR}/${VAR}_all_info.csv
MESH=${DDIR}/mesh_1km.csv
MODL=${ODIR}/checkpoint.pt
'''
#TRAIN
python  ${SRCS}/train_deepspeed.py --cuda --deepspeed_config ${CONF} \
                           --epochs 2000 --outf ${ODIR} --dataf ${DATA} \
			   --infof ${INFO} --meshf ${MESH} --var ${VAR}
#TEST
python  ${SRCS}/test_deepspeed.py --outf ${ODIR} --net ${MODL} \
                                  --dataf ${DATA} --infof ${INFO} \
                                  --meshf ${MESH} --var ${VAR}
'''
python  ${SRCS}/Data_Generation.py --outf ${ODIR} --var ${VAR} --ldapsdir ${LDAPSDIR} \
                       --dainf ${DAIN} --infof ${INFO} --meshf ${MESH} --dataf ${DATA}


