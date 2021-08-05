#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 17:21:23 2017

@author: willian
"""

import logging
from os import path

'''
Classe Dicionario de uma base de patches de imagens
'''

class DictBasePatches():
    def __init__(self, base):
        self.base = base  # nomes dos arquivos das bases de atributos relacionadas        
        self.imagens = []
        self.patches = {}
    
    # Adiciona uma nova imagem na lista de imagens
    def addImagem(self, imagem, tamanho, rotulo, ids_patches):
        if path.isfile(imagem):
            self.imagens.append({"arquivo": imagem, "tamanho": tamanho, "rotulo": rotulo, "ids_patches": ids_patches})
        else:
            logging.info("O arquivo relativo a imagem deve ser uma string")
            
    # Adiciona um novo patch na lista de patches        
    def addPatch(self, numero, posx, posy, tamanho, imagem, rotulo):
        if path.isfile(imagem):
            self.patches[numero] = {"numero": numero, "tamanho": tamanho, "posx": posx, "posy": posy, "imagem": imagem, "rotulo": rotulo}
        else:
            logging.info("O arquivo passado nao existe!")
            
    # Remove um patch da base
    def delPatch(self, numero):
        if numero in self.patches:            
            # Remove da lista de patches que compoe a imagem
            for img in self.imagens:
                if numero in img['ids_patches']:
                   img['ids_patches'].remove(numero)
                   
            # Remove da lista geral de patches       
            del self.patches[numero]