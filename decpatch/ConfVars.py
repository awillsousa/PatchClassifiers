#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 22:20:49 2017

@author: willian
"""

from keras import backend as K


BATCH_SIZE = 25
NUM_CLASSES = 2
SUBIMG_SIZE = 128
SUBIMG_DELTA = 80
PTX_SIZE = 64
PTX_DELTA = 16

INPUT_SHAPE = (PTX_SIZE, PTX_SIZE, 3)
if K.image_data_format() == 'channels_first':
   INPUT_SHAPE = (3, PTX_SIZE, PTX_SIZE)

        
    


# Formato padrao de arquivos de saida das bases de atributos
FMT_SVMLIGHT = 'svmlight'

# Formatos de imagem suportados
TIPO_IMGS = ['*.png', '*.jpeg', '*.jpg', '*.tiff', '*.bmp']

# Bases suportadas
BASES = ['breakhis', 'dtd', 'especies', 'simpsons']
BASE_DEFAULT=BASES[0]

# Indica os metodos de extracao de atributos disponiveis
METODOS = ['pftas', 'lbp', 'glcm', 'cnn']
METODO = 'pftas'

# Indica os tipos de extracao disponiveis
TIPOS = ['janela', 'ppi', 'intervalo', 'quadtree', 'randomico'] 
TIPO = 'janela'

# Tamanho padrao dos patches (quadrados)
TAM_PATCH = 64

# Quantidade padrao de patches por imagem a serem extraidos
PPI = 70

# Percentual de sobreposicao padrão dos patches
PERC_SOBREPOE = 0

# Incremento padrao nos tamanhos dos patches para extracao
# de intervalos de patches
INCR = 16

# Altura padrao para extracao do tipo quadtree
ALTURA = 3

# Nivel padrao para extracao do tipo quadtree
NIVEL = 3

# Parametros padrao para LBP
PARAMS_LBP = {'raio':3, 'pontos':24}

# Parametros padrao para GLCM
PARAMS_GLCM = {'distancia': 1}

# Rotulos das classes 
ROTULOS_CLASSES = {'B':0, 'M':1}    # classes das imagens de cancer { B - benigno, M - maligno }
ROTULOS_SUBCLASSES = {'A':0, 'F':1, 'TA':2, 'PT':3, 'DC':4, 'LC':5, 'MC':6, 'PC':7}  # subclasses das imagens de cancer

ESPECIES_CLASSES = {'Hardwood':0, 'Softwood':1}
                     
LABELS = {'cancer': ['benigno', 'maligno'],
          'especies': ['Hardwood', 'Softwood']}
                     
                     
                     
# Classificadores 
CLFS = ['knn', 'svm', 'dt', 'rf']
CLF = 'rf'                     

# Caminho modelo Imagenet pre-treinado
CNN_PATH='models/classify_image_graph_def.pb'