#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 22:12:57 2017

@author: willian

"""

import mahotas as mh
import logging
from os import path
from Imagem import Imagem
from matplotlib import pyplot as plt
from matplotlib import patches 

'''
Patch extraido de uma imagem
'''
class Patch():
    '''
        num - identificacao do patch
        arq_origem - arquivo de imagem de onde o patch foi extraido
        tamanho - tupla contendo o tamanho
        posicao - tupla indicando a posicao do patch em relacao a imagem
        dados - matriz contendo os dados (pixels)
        rotulo - rotulo do patch, para fins de classificacao
        rgb - indica se a imagem e rgb ou escala de cinza (False)
    '''
    def __init__(self, num, arq_origem, tamanho, posicao, dados, rotulo, rgb=True):
        self.num = num
        self.imagem = arq_origem
        self.dados = dados
        self.tamanho = tamanho
        self.rgb = rgb
        self.rotulo = rotulo
        self.posicao = posicao        
        self.arquivo = ""       
    '''
     Grava o patch como arquivo de imagem
    '''            
    def toArq(self, caminho, formato='png'): 
        
        nome_img = path.basename(str(self.imagem))
        nfname = ('%s/%s-%03d.%s' % (caminho, nome_img.split('.')[0] , self.num, formato))
        mh.imsave(nfname, self.dados)
        self.arquivo = nfname
    
    # Exibe o patch    
    def show(self):
        if (self.dados):
            plt.imshow(self.dados)
            plt.show()
            plt.clf()
        else:
            logging.info("Imagem não carregada!")
            
    # Exibe o patch na sua imagem de origem
    def showOnImg(self):
        imagem_orig = Imagem(self.imagem)
        im = imagem_orig.dados
        
        # Create figure and axes
        fig,ax = plt.subplots(1)
        
        # Display the image
        ax.imshow(im)
        
        # Create a Rectangle patch
        rect = patches.Rectangle(self.posicao,*self.tamanho,linewidth=1,edgecolor='g',facecolor='none')
        
        # Add the patch to the Axes
        ax.add_patch(rect)
        
        plt.show()
        plt.clf()