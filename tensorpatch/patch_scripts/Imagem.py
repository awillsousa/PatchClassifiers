#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 22:12:57 2017

@author: willian
"""


import sys
import ConfVars
import mahotas as mh
from os import path
import numpy as np
import logging
from matplotlib import pyplot as plt


'''
Imagem
'''
class Imagem():
    
    def __init__(self, arquivo, rgb=True):
        self.formato = self.getFormato(arquivo)
        self.arquivo = arquivo
        self.rgb = rgb
        
        if not path.isfile(arquivo):
            sys.exit("Erro ao carregar a imagem")
        
        self.load(arquivo)
        self.tamanho = self.dados.shape
        self.rotulo = ConfVars.ROTULOS_CLASSES[self.getClasse()[0]]
        self.idxpatches = []
        
    # Carrega o arquivo da imagem    
    def load(self, arquivo, rgb=True):        
        if rgb:
            self.dados = mh.imread(arquivo)
        else:
            self.dados = mh.imread(arquivo, as_grey=True)
            self.dados = self.dados.astype(np.uint8)
            
    # Retorna uma imagem rgb em escala de cinza
    def getCinza(self):
        return(mh.colors.rgb2grey(self.dados))
    
    # Retorna o formato da imagem (extensao do arquivo)    
    def getFormato(self, arquivo):
        return (path.basename(arquivo.upper()).split('.')[-1])
    
    # Retorna a classe a subclasse da imagem 
    def getClasse(self):
        info_arquivo =  str(self.arquivo[self.arquivo.rfind("/")+1:]).split('_')        
        
        if info_arquivo:
            classe = info_arquivo[1]            
            subclasse = info_arquivo[2].split('-')[0]
        else:
            logging.info("Problema ao recuperar o rotulo/classe da imagem.")
            
        return (classe,subclasse)
    
    # Exibe a imagem passada
    def show(self):        
        try:
            plt.imshow(self.dados)
            plt.show()
            plt.clf()
        except Exception as e:
            logging.info("Erro ao exibir a imagem {0}".format(str(e)))