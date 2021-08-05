#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 22:12:57 2017

@author: willian
"""

import json
import sys
import logging
import Imagem
import Patch
import ConfVars
import mahotas as mh
import numpy as np
from os import path
from cnn import Inception
from cv2 import cvtColor,COLOR_RGB2GRAY


'''
Classe de descritor de atributos
'''    
class Descritor:
    def __init__(self, tipo):
        self.tipo = tipo
        self.cnn = None
        
    def executa(self, imagem, params=None):
        if self.tipo == "lbp":
            return(self.executaLBP(imagem, **ConfVars.PARAMS_LBP))
        elif self.tipo == "pftas":
            return(self.executaPFTAS(imagem))
        elif self.tipo == "glcm":
            return(self.executaGLCM(imagem, **ConfVars.PARAMS_GLCM))
        elif self.tipo == "cnn":
            return(self.executaCNN(imagem))
        
    
    # Executa extracao de atritubos LBP de uma imagem 
    def executaLBP(self, imagem, raio, pontos):
        logging.info("\t\tExecutando descritor sobre a imagem {0}".format(imagem.arquivo))
        imgdata = imagem.dados
        if imagem.rgb:
            imgdata = mh.colors.rgb2grey(imgdata)
            
        return (mh.features.lbp(imgdata, radius=3, points=24))
    
    
    # Executa a extracao de atributos PFTAS de uma imagem 
    def executaPFTAS(self, imagem):
        logging.info("\t\tExecutando descritor sobre a imagem {0}".format(imagem.arquivo))
        return (mh.features.pftas(imagem.dados))
    
    # Executa a extracao de atributos GLCM de uma imagem
    def executaGLCM(self, imagem, params):
        imgdata = imagem.data
        glcm_medias = mh.features.haralick(mh.colors.rgb2gray(imgdata, dtype=np.uint8), return_mean=True)  
        
        return (glcm_medias)
    
    # Executa a extracao de atributos utilizando Inception (CNN)
    def executaCNN(self, imagem):
        # Inicializa Inception
        if not self.cnn:
            layer="pool_3:0"             
            self.cnn = Inception(ConfVars.CNN_PATH,layer)
            
        # Extrai os atributos da imagem utilizando a CNN
        imgdata = cvtColor( imagem.dados, COLOR_RGB2GRAY )
        atribs_cnn = self.cnn.describe(imgdata)
        
        return(atribs_cnn)