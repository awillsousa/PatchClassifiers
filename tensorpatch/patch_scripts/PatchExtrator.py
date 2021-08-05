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
from os import path


'''
Realiza a extracao de patches a partir de uma imagem passada 
'''
class PatchExtrator:
    def __init__(self, conf):        
        self.extracao = Extracao(conf)
        self.descritor = Descritor(conf)
    
    # Retorna uma lista de patches da imagem passada    
    def getPatches(self, imagem):
        return (self.extracao.executa(imagem))
        
    # Aplica um descritor sobre uma lista de patches
    def getAtribs(self, patches):
        atributos = []
        for patch in patches:
            atributos.append(self.descritor.executa(patch))
            
        return (atributos)

'''
Classe de extracao
'''
class Extracao:
    def __init__(self, conf):
        self.conf = conf

    # Chamada generica para a execucao da extracao de patches        
    def executa(self, imagem):
        if self.conf['descricao'] == "janela":
            #return (self.executa_jd(imagem, self.conf['parametros']))
            return (self.executaJD(imagem, **self.conf['parametros']))
        elif self.conf['descricao'] == "ppi":
            #return (self.executa_ppi(imagem, self.conf['parametros']))
            return (self.executaPPI(imagem, **self.conf['parametros']))
        elif self.conf['descricao'] == "fixo":
            #return (self.executa_fixo(imagem, self.conf['parametros']))
            return (self.executaFixo(imagem, **self.conf['parametros']))
        elif self.conf['descricao'] == "intervalo":
            #return (self.executa_intervalo(imagem, self.conf['parametros']))
            return (self.executaIntervalo(imagem, **self.conf['parametros']))
        elif self.conf['descricao'] == "quadtree":
            #return (self.executa_quadtree(imagem, self.conf['parametros']))
            return (self.executaQuadtree(imagem, **self.conf['parametros']))
        elif self.conf['descricao'] == "randomico":
            #return (self.executa_randomico(imagem, self.conf['parametros']))
            return (self.executaRandomico(imagem, **self.conf['parametros']))
    
    
    # Executa extracao de patches utilizando um esquema de janela deslizante
    # imagem - objeto do tipo Imagem
    # tamanho - tamanho da janela/patch
    # sobrepoe - percentual de sobreposicao dos patches
    def executaJD(self, imagem, tamanho, sobrepoe):
        pass
    
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
    
    # Executa a extracao de um conjunto de patches a partir de uma imagem
    # utilizando um esquema de quadtree: a imagem é divida em 4 partes na primeira
    # iteração, em 16 partes na segunda, e assim segue até a altura ou nivel da quadtree 
    # passado
    # imagem - objeto do tipo Imagem
    # altura - altura da quadtree
    # nivel - nivel da quadtree
    def executaQuadtree(self, imagem, altura=None, nivel=None):
        pass
    
    # Executa a extracao de patches aleatorios de uma imagem
    # imagem - objeto do tipo Imagem
    # tamanho - tamanho do patch quadrado
    # qtde_patches - quantidade de patches
    # sobrepoe - percentual de sobreposicao dos patches
    def executaRandomico(self, imagem, tamanho, qtde_patches, sobrepoe):
        pass
    
    
'''
Classe de descritor de atributos
'''    
class Descritor:
    def __init__(self, tipo):
        self.tipo = tipo
        
    def executa(self, imagem, params):
        if self.tipo == "lbp":
            return(self.executaLBP(imagem, **ConfVars.PARAMS_LBP))
        elif self.tipo == "pftas":
            return(self.executaPFTAS(imagem))
        if self.tipo == "glcm":
            return(self.executaGLCM(imagem, **ConfVars.PARAMS_GLCM))
    
    # Executa extracao de atritubos LBP de uma imagem 
    def executaLBP(self, imagem, raio, pontos):
        pass
    
    # Executa a extracao de atributos PFTAS de uma imagem 
    def executaPFTAS(self, imagem):
        pass
    
    # Executa a extracao de atributos GLCM de uma imagem
    def executaGLCM(self, imagem, params):
        pass
    