#!/bin/bash

python dyngesn-model.py --dataset $1 --device $2 | tee -a $1.txt
python e2e-models-1h.py --model dcrnn --dataset $1 --device $2 | tee -a $1.txt
python e2e-models-1h.py --model gconvgru --dataset $1 --device $2 | tee -a $1.txt
python e2e-models-2h.py --model gconvlstm --dataset $1 --device $2 | tee -a $1.txt
python e2e-models-2h.py --model gclstm --dataset $1 --device $2 | tee -a $1.txt
python e2e-models-dygrae.py --dataset $1 --device $2 | tee -a $1.txt
python e2e-models-0h.py --model egcnh --dataset $1 --device $2 | tee -a $1.txt
python e2e-models-0h.py --model egcno --dataset $1 --device $2 | tee -a $1.txt
python e2e-models-a3tgcn.py --dataset $1 --device $2 | tee -a $1.txt
python e2e-models-1h.py --model tgcn --dataset $1 --device $2 | tee -a $1.txt
python e2e-models-0h.py --model mpnnlstm --dataset $1 --device $2 | tee -a $1.txt
