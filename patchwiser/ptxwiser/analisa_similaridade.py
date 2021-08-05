#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 02:42:54 2017

@author: willian

Avalia uma proposta de reducao de dados baseada em similaridade dos patches.
Busca patches semelhantes em cada imagem, de maneira a reduzir aqueles semelhantes
deixando apenas um exemplar. 

Em seguida busca reduzir patches semelhantes, dentro da mesma classe. Deixando apenas
um representante.

Por ultimo, busca patches similares entre as classes. De maneira que os patches com alto
grau de similaridade serão eliminados
"""
import pickle
import numpy as np
import networkx as nx
import string
import matplotlib.pyplot as plt
from Display import Display
from Imagem import Imagem
from BaseAtributos import BaseAtributos
from DictBasePatches import DictBasePatches
from sklearn.preprocessing import normalize
#from sklearn.metrics.pairwise import euclidean_distances
from networkx.drawing.nx_agraph import graphviz_layout
from scipy.spatial import distance
from mahotas import imread
from mahotas.colors import rgb2grey
from mpl_toolkits.axes_grid1 import AxesGrid
import logging

import warnings
warnings.filterwarnings("ignore")

ID_G = 0
GRAVA = True
SCALA = 1

# Configura para o log ser exibido na tela
console = logging.StreamHandler()
console.setLevel(logging.INFO)            
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)   
logging.getLogger('').addHandler(console)  

def exibe_graph_dist(adjacency_matrix, mylabels=None):
    plt.figure(figsize=(12,12)) 
    plt.axis('equal')
    rows, cols = np.where(adjacency_matrix != 0)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    
    nx.draw(gr, pos=graphviz_layout(gr), node_size=50, with_labels=False)
    '''
    if mylabels:
        nx.draw(gr, pos=graphviz_layout(gr), node_size=50, labels=mylabels, with_labels=True) #, edge_color=colors,width=4,edge_cmap=plt.cm.Blues)
    else:
        nx.draw(gr, pos=graphviz_layout(gr), node_size=50, with_labels=False) #, edge_color=colors,width=4,edge_cmap=plt.cm.Blues)
    '''

def exibe_patches(img, tam_patch, rgb=True):
    
    import matplotlib.pyplot as plt
    import matplotlib.ticker as plticker
    try:
        from PIL import Image
    except ImportError:
        import Image
        
    image = img    
    tam_x = int(img.shape[0]/tam_patch)
    tam_y = int(img.shape[1]/tam_patch)
    
    # Set up figure
    fig,axes = plt.subplots(tam_x,tam_y)
    ax=fig.add_subplot(111)
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    # Remove whitespace from around the image
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    
    # Set the gridding interval: here we use the major tick interval
    myInterval=tam_patch
    loc = plticker.MultipleLocator(base=myInterval)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    
    # Add the grid
    ax.grid(which='major', axis='both', linestyle='-')
    
    
    # Add the image
    ax.imshow(image)
    
    # Find number of gridsquares in x and y direction
    nx=abs(int(float(ax.get_xlim()[1]-ax.get_xlim()[0])/float(myInterval)))
    ny=abs(int(float(ax.get_ylim()[1]-ax.get_ylim()[0])/float(myInterval)))
    
    # Add some labels to the gridsquares
    for j in range(ny):
        y=myInterval/2+j*myInterval
        for i in range(nx):
            x=myInterval/2.+float(i)*myInterval
            ax.text(x,y,'{:d}'.format(i+j*nx),color='r',ha='center',va='center', fontsize='18')


# Carrega a base 
DIR_BASE="/home/willian/bases/execs/cancer/fold1/train/400X/"
bases = {'cnn'   : {'base': "f1-400X-tr-p64-d100-cnn.svm", 'dict': "f1-400X-tr-p64-d100-cnn.ppi", 'tam_atribs':2048},
         'pftas' : {'base': "f1-400X-tr-p64-d100-pftas.svm", 'dict': "f1-400X-tr-p64-d100-pftas.ppi", 'tam_atribs':168}}
'''
         'lbp'   : {'base': "min-tr-p64-d100-lbp.svm", 'dict': "min-tr-p64-d100-lbp.ppi", 'tam_atribs':8},
         'glcm'  : {'base': "min-tr-p64-d100-glcm.svm", 'dict': "min-tr-p64-d100-glcm.ppi", 'tam_atribs':14},
         'sift'  : {'base': "min-tr-p64-d100-sift.svm", 'dict': "min-tr-p64-d100-sift.ppi", 'tam_atribs':128}}
'''
for DESCRITOR,ARQ in bases.items():

    ARQ_BASE = ARQ['base']
    DICT_BASE = ARQ['dict']
    TAM_ATRIBS = ARQ['tam_atribs']
    
    base_atribs = BaseAtributos(DIR_BASE+ARQ_BASE, tam_atribs=TAM_ATRIBS, usa_qid=True)
    base_atribs.carregaArq()
    with open(DIR_BASE+DICT_BASE, "rb" ) as arq_dict:
        dict_base = pickle.load(arq_dict)
    
    #cemrot = base_atribs.rotulos
    #cemqid = base_atribs.qids
    cemimg = dict_base.imagens
    
    ## FASE 1
    ## Efetua os calculos de distancia apenas sobre patches da propria imagem
    '''
    distancias = []
    for img in cemimg:
        idxs_ptxs = img["ids_patches"]
        linhas = len(idxs_ptxs)
        #print("Tipo da matriz de atributos: {0} Tamanho: {1}".format(type(base_atribs.atributos), base_atribs.atributos.shape))
        #print("Tamanho indice: {0}".format(len(idxs_ptxs)))    
    
        atribs_img = base_atribs.atributos[idxs_ptxs, :].toarray()        
        distancia = distance.cdist(atribs_img, atribs_img, 'sqeuclidean')
        normalize(distancia, copy=False)
        print("Caminho imagem: {0}".format(img['arquivo']))
        
        # Exibe os patches da imagem original
        imagem = Imagem(img['arquivo'])
        imagem.idxpatches = idxs_ptxs
        
	exibe_patches(imagem.dados,64,rgb=False)
        
	if GRAVA:        
            ID_G += 1
            plt.savefig("{0}fig-{1}-{2}-patches.png".format("./plots/", ID_G, DESCRITOR))
        else:    
            plt.show()
        plt.clf()
        
        # Exibe a distribuição das distancias
        plt.hist(distancia, bins='auto')  # arguments are passed to np.histogram
        plt.title("Distribuicao de Distancias na Imagem-{0}".format(DESCRITOR))
        if GRAVA:        
            plt.savefig("{0}fig-{1}-{2}-histo.png".format("./plots/", ID_G, DESCRITOR))
        else:
            plt.show()
        plt.clf()
        
        # Exibe o grafo de distancia dos patches
        exibe_graph_dist(distancia, {i:str(i) for i in range(len(imagem.idxpatches))})
        if GRAVA:        
            plt.savefig("{0}fig-{1}-{2}-dists.png".format("./plots/", ID_G, DESCRITOR))
        else:
            plt.show()
        plt.clf()
    '''
    ## FASE 2
    ## Agrupa os pathes pela sua classe e busca similaridade intraclasse
    atribs_b = []   # patches benignos
    atribs_m = []   # patches malignos
    idx_ptxs_b = []
    idx_ptxs_m = []
    
    for img in cemimg:
        idxs_ptxs = img["ids_patches"]
        
        if img['rotulo'] == 0:
            idx_ptxs_b += idxs_ptxs
        elif img['rotulo'] == 1:
            idx_ptxs_m += idxs_ptxs
    
    atribs_b = base_atribs.atributos[idx_ptxs_b, :].toarray()
    
    distancia_b = distance.cdist(atribs_b, atribs_b, 'sqeuclidean')
    normalize(distancia_b, copy=False)
        
    # Exibe a distribuição das distancias
    plt.hist(distancia_b, bins='auto')  # arguments are passed to np.histogram
    plt.title("Distribuicao de Distancias Intraclasse (Benignos)-{0}".format(DESCRITOR))
    if GRAVA:        
        plt.savefig("{0}fig-{1}-{2}-histo.png".format("./plots/", 'BENIGNOS',DESCRITOR))
    else:
        plt.show()
    plt.clf()
    
    # Exibe o grafo de distancia dos patches
    exibe_graph_dist(distancia_b, {i:str(i) for i in range(len(idx_ptxs_b))})
    if GRAVA:        
        plt.savefig("{0}fig-{1}-{2}-dists.png".format("./plots/", 'BENIGNOS',DESCRITOR))
    else:
        plt.show()
    plt.clf()
    
    del distancia_b
    
    atribs_m = base_atribs.atributos[idx_ptxs_m, :].toarray()
    
    distancia_m = distance.cdist(atribs_m, atribs_m, 'sqeuclidean')
    normalize(distancia_m, copy=False)
        
    # Exibe a distribuição das distancias
    plt.hist(distancia_m, bins='auto')  # arguments are passed to np.histogram
    plt.title("Distribuicao de Distancias Intraclasse (Malignos)-{0}".format(DESCRITOR))
    if GRAVA:        
        plt.savefig("{0}fig-{1}-{2}-histo.png".format("./plots/", 'MALIGNOS',DESCRITOR))
    else:
        plt.show()
    plt.clf()
        
    # Exibe o grafo de distancia dos patches
    exibe_graph_dist(distancia_m, {i:str(i) for i in  range(len(idx_ptxs_m))})
    if GRAVA:        
        plt.savefig("{0}fig-{1}-{2}-dists.png".format("./plots/", 'MALIGNOS',DESCRITOR))
    else:
        plt.show()
    plt.clf()
    
    del distancia_m
    
    ## FASE 3
    ## Busca similaridade interclasses
    distancia = distance.cdist(atribs_m, atribs_b, 'sqeuclidean')
    normalize(distancia, copy=False)
        
    # Exibe a distribuição das distancias
    plt.hist(distancia, bins='auto')  # arguments are passed to np.histogram
    plt.title("Distribuicao de Distancias Intraclasse (Malignos X Benignos)-{0}".format(DESCRITOR))
    if GRAVA:        
        plt.savefig("{0}fig-{1}-{2}-histo.png".format("./plots/", 'MxB',DESCRITOR))    
    else:
        plt.show()
    plt.clf()
        
    # Exibe o grafo de distancia dos patches
    exibe_graph_dist(distancia)#, {i:str(i) for i in range(len(idx_ptxs_m)+len(idx_ptxs_b))})
    if GRAVA:        
        plt.savefig("{0}fig-{1}-{2}-dists.png".format("./plots/", 'MxB',DESCRITOR))
    else:
        plt.show()
    plt.clf()
