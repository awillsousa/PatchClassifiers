#!/bin/bash

# Executa a extracao

python /home/willian/dev/PatchWiser/npatches.py -T "/home/willian/basesML/bases_cancer/fold1/treino/" -t "/home/willian/basesML/bases_cancer/fold1/teste/" -C "dt,svm" -n 4
python /home/willian/dev/PatchWiser/npatches.py -T "/home/willian/basesML/bases_cancer/fold1/treino/" -t "/home/willian/basesML/bases_cancer/fold1/teste/" -C "dt,svm" -n 3
python /home/willian/dev/PatchWiser/npatches.py -T "/home/willian/basesML/bases_cancer/fold1/treino/" -t "/home/willian/basesML/bases_cancer/fold1/teste/" -C "dt,svm" -n 2
python /home/willian/dev/PatchWiser/npatches.py -T "/home/willian/basesML/bases_cancer/fold1/treino/" -t "/home/willian/basesML/bases_cancer/fold1/teste/" -C "dt,svm" -n 1

