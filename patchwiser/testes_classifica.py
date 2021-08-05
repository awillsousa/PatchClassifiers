# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 16:24:26 2016

@author: antoniosousa
Script de teste das funcoes do programa de extracao (extrai.py)

"""
import cv2
import unittest 
import helper
import extrai as ext
import classifica as clfc
import arquivos as arq
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from math import sqrt
       

class TestClassificaMethods(unittest.TestCase): 
    
    def test_load_csv(self):
        dados = helper.load_csv("/home/willian/bases/min_treino/base_pftas_patches64x64.ppi")        
        imagens = clfc.dicionario_imgs(dados)
        #print(str(imagens))
        self.assertEqual(len(imagens)>0, True, "Dados recuperados do arquivo csv")
    
    def test_classifica_img(self):
        #base_tr = "/home/willian/basesML/bases_cancer/min_treino/base_pftas_patches64x64.svm"
        #base_ts = "/home/willian/basesML/bases_cancer/min_teste/base_pftas_patches64x64.svm"
        #arq_ppi = "/home/willian/basesML/bases_cancer/min_teste/base_pftas_patches64x64.ppi"
        base_tr = "/home/willian/bases/min_treino/base_pftas_patches64x64.svm"
        base_ts = "/home/willian/bases/min_teste/base_pftas_patches64x64.svm"
        arq_ppi = "/home/willian/bases/min_teste/base_pftas_patches64x64.ppi"
        clf = clfc.get_clf("svm")
        atrib_tr, rotulos_tr = load_svmlight_file(base_tr, dtype=np.float32,n_features=162) 
        atrib_ts, rotulos_ts = load_svmlight_file(base_ts, dtype=np.float32, n_features=162)
        
        clf.fit(atrib_tr, rotulos_tr)
        imagens = clfc.dicionario_imgs(helper.load_csv(arq_ppi))
        
        r_tst = []
        r_pred = []
        idx1 = 0    # posicao inicial dos atributos da imagem
        idx2 = 0    # posicao final dos atributos da imagem
        #num_ppi = imagens[0]['total']     
        #print(str(imagens))
        for imagem in imagens:            
            idx2 = imagem['ppi']            
            if idx2 > 0:
                idx2 += idx1  # limite superior da fatia                
                atribs_img = atrib_ts[idx1:idx2]                 
                tst,pred = clfc.classifica_img(imagem['arquivo'], clf, atribs_img)
                r_tst.append(tst)
                r_pred.append(pred)                                
                idx1 = idx2            
        
        print(str(r_tst))
        print(str(r_pred))
        self.assertEqual(len(r_tst) == len(r_pred), True, "Classificação de imagem OK")
        
# Programa principal
if __name__ == '__main__':
    unittest.main()
    