#!/bin/bash
MAG=$1
PROB=$2

(nohup python classifica.py -T ~/folds/fold1/train/$MAG/ -t ~/folds/fold1/test/$MAG/ -m pftas -C dt -s 64 -l fold1-$MAG-pftas-dt-proba$PROB &) ; (nohup python classifica.py -T ~/folds/fold2/train/$MAG/ -t ~/folds/fold2/test/$MAG/ -m pftas -C dt -s 64 -l fold2-$MAG-pftas-dt-proba$PROB &) ; (nohup python classifica.py -T ~/folds/fold3/train/$MAG/ -t ~/folds/fold3/test/$MAG/ -m pftas -C dt -s 64 -l fold3-$MAG-pftas-dt-proba$PROB &) ; (nohup python classifica.py -T ~/folds/fold4/train/$MAG/ -t ~/folds/fold4/test/$MAG/ -m pftas -C dt -s 64 -l fold4-$MAG-pftas-dt-proba$PROB &) ; (nohup python classifica.py -T ~/folds/fold5/train/$MAG/ -t ~/folds/fold5/test/$MAG/ -m pftas -C dt -s 64 -l fold5-$MAG-pftas-dt-proba$PROB &)
