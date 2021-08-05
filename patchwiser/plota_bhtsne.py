#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 18:45:54 2017

@author: willian

Arquivo utilizado para quando se deseja plotar apenas uma base para avaliação
Utilizando apenas para testes. Não há necessidade de mantê-lo atualizado. 
Muitas coisas podem estar atrasadas ou incorretas nesse arquivo.
"""

import numpy as np
import bhtsne
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt


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


arq_tr = base+"min_treino/base_pftas_ptx64x64.svm"
plota_bhtsne(arq_tr, "min-treino")
        

