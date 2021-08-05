#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 01:40:52 2017

@author: willian

Gera uma visualizacao das bases utilizando BHtSNE

"""
import pickle
from BaseAtributos import BaseAtributos
from DictBasePatches import DictBasePatches
from Display import Display

# Carrega a base 
DIR_BASE="/home/willian/bases/execs/cancer/fold1/train/400X/"
ARQ_BASE="f1-400X-p64-d100-tr-pftas.svm"
ARQ_BASE_M="f1-400X-p64-d100-tr-pftas-0.svm"
ARQ_BASE_B="f1-400X-p64-d100-tr-pftas-1.svm"
DICT_BASE="f1-400X-p64-d100-tr-pftas.ppi"

base_atribs = BaseAtributos(DIR_BASE+ARQ_BASE, tam_atribs=162)
base_atribs.carregaArq()
dict_base = pickle.load(open(DIR_BASE+DICT_BASE, "rb" ))

base_atribs_M = BaseAtributos(DIR_BASE+ARQ_BASE_M, tam_atribs=162)
base_atribs_M.carregaArq()

base_atribs_B = BaseAtributos(DIR_BASE+ARQ_BASE_B, tam_atribs=162)
base_atribs_B.carregaArq()

# Plota a base integral
Display.visualiza_bhtsne(base={'data':base_atribs.atributos, 'labels': base_atribs.rotulos}, arquivo="f1-400X",texto="Base Integral")
Display.visualiza_bhtsne(base={'data':base_atribs_M.atributos, 'labels': base_atribs_M.rotulos}, arquivo="M-f1-400X",texto="Base Classe 0")
Display.visualiza_bhtsne(base={'data':base_atribs_B.atributos, 'labels': base_atribs_B.rotulos}, arquivo="B-f1-400X",texto="Base Classe 1")






