#!/usr/bin/env bash

#DATA=images2both
#DATA=kinetics7002both
DATA=kinetics7002others

NET=r2plus1d_18
EPOCHS=150
LR=1e-3

PRE="--nopretrained"

#WEIGHTS="pretrained_on_SUN397"
WEIGHTS="none"


SAVEPATH="/workplace/"
python3 main.py --dataset ${DATA} --save_path ${SAVEPATH} --n_epochs ${EPOCHS} --lr ${LR} --weights ${WEIGHTS} ${PRE}
