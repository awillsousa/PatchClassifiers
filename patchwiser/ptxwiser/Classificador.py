#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 19:22:06 2017

@author: willian

Realiza a classificacao de uma base de imagens baseado nos atributos dos seus patches
"""
import sys
import pickle
import logging
import numpy as np
from os import path
from time import time
from BaseAtributos import BaseAtributos
from DictBasePatches import DictBasePatches
from Display import Display
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve, auc



class Classificador():
    
    CLFS = {'knn':('KNN', KNeighborsClassifier(3)), 
                'svm':('SVM', SVC(gamma=0.5, C=32, cache_size=250, probability=True)),
                'dt':('DT', DecisionTreeClassifier(max_depth=5)),
                'rf':('RF', RandomForestClassifier(max_features=0.1, n_estimators=500, min_samples_leaf=0.01, n_jobs=3))
                }
    
    def __init__(self, conf):
        if conf is None:
          sys.exit("Configuracao ausente.")
        else:
            self.conf = conf
            
        self.clf = None
        self.ds_clf = None
        self.setClf(self.conf['clf'])
        
        self.arqtr = self.conf['base_treino']        
        self.arqts = self.conf['base_teste']
        
        # Dados dos patches da base de treino
        self.dicttr = None
        self.basetr = BaseAtributos(self.arqtr, tam_atribs=2048)
        self.basetr.carregaArq()
        
        # Dados dos patchs da base de teste (a classificar)
        self.dictts = None
        self.basets = BaseAtributos(self.arqts, tam_atribs=2048)
        self.basets.carregaArq()
        
        # Valores dos resultados de classificação
        self.tempo_medio_img = 0
        self.desvio_tempo_imgs = 0
        self.matriz_confusao = None
        self.taxa_clf = 0
        self.auc = None
        self.fpr = 0.0
        self.tpr = 0.0
        self.tempoTotal = 0
        self.totalImgs = 0
        self.totalPatches = 0
        
    
    # Executa o processo de classificação
    def executa(self):
        self.carregaDicts()
        self.classifica()
        self.gravaResultados()
        
        
    # Carrega as base de treino e de teste, alem dos dicionarios de 
    # cada uma das bases    
    def carregaDicts(self):
        logging.info("Carregando dicionarios das bases...")
        try:
            # Carrega o dicionario da base de treino
            arq_dict_tr = self.arqtr.replace(".svm", ".ppi")
            self.dicttr = pickle.load(open( arq_dict_tr, "rb" ))
            
            # Carrega o dicionario da base de teste
            arq_dict_ts = self.arqts.replace(".svm", ".ppi")
            self.dictts = pickle.load(open( arq_dict_ts, "rb" ))
            
        except Exception as e:
            logging.info("Erro durante a carga das bases: {0}".format(str(e)))
    
    def setClf(self, clf):
        c = Classificador.CLFS[clf]    
        self.clf = c[1]
        self.ds_clf = c[0]
     
    # Executa a classificacao das bases            
    def classifica(self):
        logging.info("Executando classificação...")
        
        inicio = time() 
        r_tst = []       # lista dos rotulos reais das imagens
        r_pred = []      # lista dos rotulos predito das imagens
        probs_imgs = []  # lista dos valores de probabilidades das imagens
        tempos_imgs = []  # tempos de classificacao das imagens
        total_erros = 0   # total de patches classificados incorretamente   
        
        # Treina o classificador
        logging.info("Treinando classificador...")
        self.clf.fit(self.basetr.atributos, self.basetr.rotulos)            
        self.totalImgs = len(self.dictts.imagens)
             
        for imagem in self.dictts.imagens: 
            t0_imagem = time() 
            
            idxs_patches = imagem['ids_patches']
            self.totalPatches += len(idxs_patches)
            #patches = self.dictts.patches[idxs_patches]
            atribs_patches = self.basets.atributos[idxs_patches]
            rots_patches = self.basets.rotulos[idxs_patches]
            logging.info("Classificando imagem {0}".format(imagem['arquivo']))
            label_real, label_pred, num_erros, prob = self.__classificaImagem(imagem, atribs_patches, rots_patches)
            
            r_tst.append(label_real)
            r_pred.append(label_pred)
            total_erros += num_erros
            probs_imgs.append(prob)
            
            tempos_imgs.append(round(time()-t0_imagem,3))    
            logging.info("Tempo classificação imagem: " + str(tempos_imgs[-1]))
            
        # Loga estatisticas de tempo por imagem
        self.tempo_medio_img = np.mean(tempos_imgs)
        self.desvio_tempo_imgs = np.std(tempos_imgs)
        
         # cria as matrizes de confusao
        self.matriz_confusao = confusion_matrix(r_tst, r_pred)
        
        # calcula a taxa de classificacao
        r_pred = np.asarray(r_pred)
        r_tst = np.asarray(r_tst)
        self.taxa_clf = np.mean(r_pred.ravel() == r_tst.ravel()) * 100
        
        # Calcula curva ROC/AUC        
        probas_ = np.asarray(probs_imgs)        
        self.fpr, self.tpr, thresholds = roc_curve(r_tst.ravel(), probas_[:, 1])
        self.auc = auc(self.fpr, self.tpr)
        
        self.tempoTotal = time()-inicio        
      
            
    def __classificaImagem(self, imagem, atribs_patches, rots_patches):
        # recupera o rotulo real da imagem
        rotulo_real = imagem['rotulo']                
                        
        preds_prob = self.clf.predict_proba(atribs_patches)     
        probs_img = np.max(preds_prob, axis=0)
            
        ls_preds = np.where(preds_prob[:,0] > preds_prob[:,1], 0, 1)           
        rotulo_pred = np.argmax(np.bincount(ls_preds))                
        errados = len([x for x in ls_preds if x != rotulo_real])
        
        return (rotulo_real, rotulo_pred, errados, probs_img)   
    
    # Grava o modelo treinado para arquivo
    def dumpModel(self, arquivo=None):
        if not arquivo:
            pickle.dump(self.clf, open(self.arqtr.replace('.svm', self.ds_clf+'.model'), "wb" ) )
        else:    
            pickle.dump(self.clf, open(arquivo, "wb" ) )
    # Carrega um modelo treinado a partir de arquivo
    def loadModel(self, arquivo=None):
        if not arquivo:
            self.clf = pickle.load(open(self.arqtr.replace('.svm', self.ds_clf+'.model'), "wb" ) )
        else:
            self.clf = pickle.load(open(arquivo, "wb" ) )
    
    # armazena resultados nos logs, plota curvas e graficos resultantes        
    def gravaResultados(self):
        logging.info("Gravando resultados...")
        
        logging.info("\nTempo medio de classificacao por imagem: {0}".format(self.tempo_medio_img))
        logging.info("\tDesvio-padrao: {0}".format(self.desvio_tempo_imgs))
        logging.info("Taxa de Classificação: {0}".format(self.taxa_clf))
        logging.info("Matriz de Confusão: {0}".format(self.matriz_confusao))
        
        # Plota a curva AUC
        # Plota as curvas do treinamento
        id_visualiz = path.basename(self.conf['log']).replace(".log","")
        Display.plot_roc(self.fpr, self.tpr, self.auc, label="", arquivo=id_visualiz, titulo="Curva ROC")            
        
        # Grava o arquivo do modelo treinado
        self.dumpModel()   	
