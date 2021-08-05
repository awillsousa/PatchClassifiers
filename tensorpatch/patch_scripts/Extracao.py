#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 22:12:57 2017

@author: willian
"""

import json
import sys
import logging
import ConfVars
from Imagem import Imagem
from Patch import Patch
import numpy as np
import sliding_window as sw
from os import path
from math import floor


'''
Classe de extracao
'''
class Extracao:
    def __init__(self, conf=None):
        self.conf = conf
        self.tamanho = None
        
    # Chamada generica para a execucao da extracao de patches        
    def executa(self, imagem):
        if not self.conf:
            logging.info("Metodo executa somente pode ser executado com uma configuracao definida.")
            return
        
        if self.conf['descricao'] == "janela":
            h = int(self.conf['parametros']['tamanho'])
            if not self.tamanho:
                self.tamanho = (h,h)
            return (self.executaJD(imagem, **self.conf['parametros']))
        elif self.conf['descricao'] == "ppi":
            h = int(self.conf['parametros']['tamanho'])
            if not self.tamanho:
                self.tamanho = (h,h)
            return (self.executaPPI(imagem, **self.conf['parametros']))
        elif self.conf['descricao'] == "fixo":
            h = int(self.conf['parametros']['tamanho'])
            if not self.tamanho:
                self.tamanho = (h,h)
            return (self.executaFixo(imagem, **self.conf['parametros']))
        elif self.conf['descricao'] == "intervalo":
            h = int(self.conf['parametros']['tamanho'])
            if not self.tamanho:
                self.tamanho = (h,h)
            return (self.executaIntervalo(imagem, **self.conf['parametros']))
        elif self.conf['descricao'] == "quadtree":
            tamX, tamY = self.__tam4_patch(*imagem.tamanho, 0,0,self.conf['parametros']['altura'])
            self.tamanho = (tamX, tamY)        
            return (self.executaQuadtree(imagem, **self.conf['parametros']))
        elif self.conf['descricao'] == "randomico":
            h = int(self.conf['parametros']['tamanho'])
            if not self.tamanho:
                self.tamanho = (h,h)
            return (self.executaRandomico(imagem, **self.conf['parametros']))
    
    # Executa extracao de patches utilizando um esquema de janela deslizante
    # imagem - objeto do tipo Imagem
    # tamanho - tamanho da janela/patch
    # sobrepoe - percentual de sobreposicao dos patches
    def executaJD(self, imagem, tamanho, sobrepoe):
        logging.info("Extraindo patches da imagem {0}".format(imagem.arquivo))
        desloca = int(tamanho*sobrepoe/100)
        patches = []
        altura,largura = imagem.tamanho[:2]      # duas primeiras dimensoes do tamanho da imagem
        
        margem_v = ((imagem.tamanho[0]-tamanho) % desloca) // 2 
        margem_h = ((imagem.tamanho[1]-tamanho) % desloca) // 2 
                                       
        i=0
        for linha in range(margem_v, altura-tamanho, desloca):            
            for coluna in range(margem_h, largura-tamanho, desloca):                
                p = imagem.dados[linha:linha+tamanho, coluna:coluna+tamanho]
                posicao = (coluna,linha)
                patches.append(Patch(i,imagem.arquivo,(tamanho,tamanho),posicao, p,imagem.rotulo))  
                i+=1
                
        logging.info("Extraidos {0} patches".format(str(len(patches))))
        
        return (patches)
    
    # Executa extracao de um numero fixo de patches por imagem
    # A imagem sera dividida em uma quantidade fixa de patches,
    # retangulares ou quadrados
    # imagem - objeto do tipo Imagem
#   # qtd_patches - quantidade de patches por imagem        
    def executaPPI(self, imagem, qtde_patches):
        pass
    
    # Executa a extracao de um conjunto de patches de tamanho fixo de uma imagem
    # incrementa esse tamanho e extrai um outro conjunto de patches. Repete o pro-
    # cesso até o tamanho_fim. 
    # imagem - objeto do tipo Imagem
    # tamanho_ini - tamanho inicial do intervalo de patches
    # tamanho_fim - tamanho final do intervalo de patches
    # incr - incremento do valor do tamanho
    def executaIntervalo(self, imagem, tamanho_ini, tamanho_fim, incr):
        pass
    
    
    # Recebe uma area e divide a mesma k vezes
    # em 4 partes iguais. Retorna o tamanho da menor divisao 
    # (0,0,x,y)    
    def __tam4_patch(self, x,y,x0=0,y0=0,k=1):
        if (k == 0):
            return x,y
            
        # calcula os valores medios
        y_m = int(floor((y - y0)/2))             
        x_m = int(floor((x - x0)/2))
        k -= 1
        
        return (self.__tam4_patch(x_m, y_m, x0, y0, k))    
    
    # Executa a extracao de um conjunto de patches a partir de uma imagem
    # utilizando um esquema de quadtree: a imagem é divida em 4 partes na primeira
    # iteração, em 16 partes na segunda, e assim segue até a altura ou nivel da quadtree 
    # passado
    # imagem - objeto do tipo Imagem
    # altura - altura da quadtree
    # nivel - nivel da quadtree
    def executaQuadtree(self, imagem, altura=None, nivel=None):
        tamX, tamY = self.__tam4_patch(*imagem.tamanho, 0,0,altura)
        #window_size = (tamX, tamY) if not(imagem.rgb) else (tamX, tamY,3)        
        #patches = [Patch(i,imagem.arquivo,tamanho,p,imagem.rotulo) for i,p in enumerate(sw.sliding_window_nd(imagem.dados, window_size, ss=step_size))]  
        patches = []
        imgH,imgW = imagem.shape[:2]
        i=0 
        for y in range(0, imgH, tamY):            
            for x in range(0, imgW, tamX):                
                p = imagem.dados[x:x+tamX, y:y+tamY]
                posicao = (x,y)
                patches.append(Patch(i,imagem.arquivo,(tamX,tamY),posicao, p,imagem.rotulo))  
                i+=1
        
        logging.info("Extraidos {0} patches".format(str(len(patches))))
                
        return (patches)
    
    
    # Executa a extracao de patches aleatorios de uma imagem
    # imagem - objeto do tipo Imagem
    # tamanho - tamanho do patch quadrado
    # qtde_patches - quantidade de patches
    # sobrepoe - percentual de sobreposicao dos patches
    def executaRandomico(self, imagem, tamanho, qtde_patches, sobrepoe):
        pass
    
