# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 13:34:10 2016
@author: antoniosousa
Testes de classificação
"""

import datetime
from os import path
from time import time
from skimage.filters import gaussian
from skimage.feature import local_binary_pattern
from optparse import OptionParser
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import cv2
import sys
import math
import numpy as np
import mahotas as mh
import extrator as ex
import arquivos as arq
import binarypattern as bp
import matplotlib.pyplot as plt
import sliding_window as sw




###############################################################################

#def main():
treino="/home/willian/basesML/bases_cancer/folds-spanhol/mkfold/fold5/train/40X/"
teste="/home/willian/basesML/bases_cancer/folds-spanhol/mkfold/fold5/test/40X/"
img_teste = "/home/willian/basesML/bases_cancer/min_treino/SOB_B_A-14-22549G-100-010.png"
#treino="/home/willian/basesML/bases_cancer/min_treino2/"    
#teste="/home/willian/basesML/bases_cancer/min_teste/"    
n=128
metodo="pftas"

#executa_extracao_n(treino, metodo, n)
#executa_classificacao_n(teste,treino,n)
imagem = mh.imread(img_teste)
classe, _ = ex.classe_arquivo(img_teste)
atrs,rots,tp_patch = ex.extrai_atrib_patches(imagem, "B", 64, "fixo", 'glcm')
glcm = mh.features.haralick(mh.colors.rgb2gray(imagem, dtype=np.uint8), return_mean=True)#np.asarray(imagem, dtype=np.uint8))

    
# Programa principal 
#if __name__ == "__main__":    
#	main()