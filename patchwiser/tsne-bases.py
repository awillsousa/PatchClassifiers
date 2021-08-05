# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 16:12:29 2017

@author: antoniosousa
"""

import pickle
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import fetch_mldata
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from numpy import float32
#%pylab inline

folds = ['fold1','fold2','fold3','fold4','fold5']
mags = ['40X', '100X', '200X', '400X']
#base="/folds/execs/bases/cancer/"
base="/home/willian/bases/execs/cancer/"
t = "/train/"
fold = folds[4]
mag = mags[3]

'''
for mag in mags:        
    for t in ["/train/", "/test/"]:
        for i,fold in enumerate(folds):
'''            
base_tr = base+fold+t+mag+"/base_pftas_patches64x64.svm"
print("Base {0}".format(base_tr))
#base_tr="Y:/bases/execs/fold5/train/40X/base_pftas_ptx64x64.svm"
#base_tr="Y:/bases/min_treino/base_pftas_ptx64x64.svm"
atrib_tr, rotulos_tr = load_svmlight_file(base_tr, dtype=float32,n_features=162)
base = {}
base["data"] = atrib_tr.toarray()
base["target"] = rotulos_tr

print(str(type(base["data"])) + " - " + str(type(base["target"])))
print(str(base["data"].shape) + " - " + str(base["target"].shape))

size = 50
N = size * size
data, target = shuffle(base["data"], base["target"], random_state=777, n_samples=int(base["target"].shape[0]/5))
#data, target = shuffle(base["data"], base["target"], random_state=777, n_samples=N)
            
data_100 = PCA(n_components=100).fit_transform(data.astype(float32) / 255)
embeddings = TSNE(init="pca", random_state=777, verbose=2, perplexity=50.0, early_exaggeration=4.0, learning_rate=500.0, n_iter=1000).fit_transform(data_100)
embeddings -= embeddings.min(axis=0)
embeddings /= embeddings.max(axis=0)

plt.rcParams["figure.figsize"] = (17, 9)
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=target)
my_colorbar = plt.colorbar(fraction=0.05, pad = 0.0125)
plt.xticks([])
plt.yticks([])
plt.savefig("plot-tSNE-{0}-{1}.png".format(fold,mag))