#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 18:45:54 2017

@author: willian

Plota todas os folds da base de cancer utilizando a implementação
de Barnes-Hut do algoritmo tSNE para visualizacao dos dados.

"""

import numpy as np
import bhtsne
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt


folds = ['fold1','fold2','fold3','fold4','fold5']
mags = ['40X', '100X', '200X', '400X']

base="/home/awsousa/execs/"

def plota_bhtsne(base, arq_plot):    
    # Carrega arquivo
    atrib_ts, rotulos_ts = load_svmlight_file(base, dtype=np.float32, n_features=162)
    
    embeddings = bhtsne.run_bh_tsne(atrib_ts.toarray(), initial_dims=atrib_ts.shape[1])
            
    plt.rcParams["figure.figsize"] = (17, 9)
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=rotulos_ts)
    my_colorbar = plt.colorbar(fraction=0.05, pad = 0.0125)
    plt.xticks([])
    plt.yticks([])
    plt.savefig("plot-BHtSNE-{0}.png".format(arq_plot))
    plt.close() 


for mag in mags:   
    for i,fold in enumerate(folds):    
        print("Magnitude {0} - Fold {1}".format(mag, fold))
        arq_ts = base+fold+"/test/"+mag+"/base_pftas_patches64x64.svm"
        arq_tr = base+fold+"/test/"+mag+"/base_pftas_patches64x64.svm"
        plota_bhtsne(arq_tr, "treino-{0}-{1}".format(mag,fold))
        plota_bhtsne(arq_ts, "teste-{0}-{1}".format(mag,fold))
        

