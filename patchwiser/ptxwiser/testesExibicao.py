#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:43:42 2017

@author: willian
"""

import pickle
from Imagem import Imagem
from Patch import Patch
from BaseAtributos import BaseAtributos
from DictBasePatches import DictBasePatches


# carrega o dicionario da base de dados
arq_dict_base = "/home/willian/bases/amostras/cancer/dest/amostra-treino-pftas.ppi"
dict_base = pickle.load(open( arq_dict_base, "rb" ))
'''
# lista imagens da base
for dict_img in dict_base.imagens:
    print(str(dict_img))
    img = Imagem(dict_img['arquivo'])
    img.show()
    break
'''    
# exibe um patch da ultima imagem  
dict_patch = dict_base.patches[9]
print(str(dict_patch)) 
tam = dict_patch['tamanho']
img = Imagem(dict_patch['imagem'])

posx = dict_patch['posx']
posy = dict_patch['posy']
rot =  dict_patch['rotulo']   
patch = Patch(10, dict_patch['imagem'], tam, (posx,posy), img.dados[posx:posx+tam[0]][posy:posy+tam[1]], rot) 
patch.showOnImg()
