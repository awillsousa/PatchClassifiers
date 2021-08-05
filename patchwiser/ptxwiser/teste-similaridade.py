#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 02:42:54 2017

@author: willian

Avalia uma proposta de reducao de dados baseada em similaridade dos patches.
Busca patches similares entre as classes. De maneira que os patches com alto
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
from sklearn.preprocessing import StandardScaler
#from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.datasets import dump_svmlight_file
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
TESTE = True


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

if TESTE:
    DIR_BASE="/bases/execs/cancer/min_treino/"
    bases = {'cnn'   : {'base': "min-tr-p64-d100-cnn.svm", 'dict': "min-tr-p64-d100-cnn.ppi", 'tam_atribs':2048}}


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
    
    ## Agrupa os pathes pela sua classe e busca similaridade intraclasse
    atribs_b = []   # patches benignos
    atribs_m = []   # patches malignos
    idx_ptxs_b = []
    idx_ptxs_m = []
    
    imgs_e_ptxs = [] # lista contendo tuplas de imagens e a suas listas de patches
    
    for img in cemimg:
        idxs_ptxs = img["ids_patches"]
        
        if img['rotulo'] == 0:
            idx_ptxs_b += idxs_ptxs
        elif img['rotulo'] == 1:
            idx_ptxs_m += idxs_ptxs
    
    print("INICIO Indices dos paches")            
    print (str(idx_ptxs_b))
    print (str(idx_ptxs_m))        
    print("FIM Indices dos paches")            
    
    atribs_b = base_atribs.atributos[idx_ptxs_b, :].toarray()
    atribs_m = base_atribs.atributos[idx_ptxs_m, :].toarray()
    
    ## FASE 3
    ## Busca similaridade interclasses
    #distancia = distance.cdist(atribs_m, atribs_b, 'sqeuclidean')
    #normalize(distancia, copy=False)
    distancia = pairwise_distances(atribs_m, atribs_b, 'euclidean', n_jobs=-1)
    normalize(distancia, copy=False)
        
    # Exibe a distribuição das distancias
    plt.hist(distancia, bins='auto')  # arguments are passed to np.histogram
    plt.title("Distribuicao de Distancias Intraclasse (Malignos X Benignos)-{0}".format(DESCRITOR))
    if GRAVA:        
        plt.savefig("{0}fig-{1}-{2}-histo.png".format("./plots/", 'MxB',DESCRITOR))    
    else:
        plt.show()
    plt.clf()
        
    '''
    A representacao exibida pelo grafo de distancias nao esta bem representado, 
    talvez devessemos utilizar uma outra biblioteca que forneca uma melhor representacao
    ou não utilizar (opção que escolhi)
    
    # Exibe o grafo de distancia dos patches    
    exibe_graph_dist(distancia)
    if GRAVA:        
        plt.savefig("{0}fig-{1}-{2}-dists.png".format("./plots/", 'MxB',DESCRITOR))
    else:
        plt.show()
    plt.clf()
    '''
    
    del_idx_b = []
    del_idx_m = []
    
    # Busca patches "proximos"/similares, porem de classes distintas
    visitados = []
    for indice, valor_dist in np.ndenumerate(distancia):
        if not((indice[0], indice[1]) in visitados) and not((indice[1], indice[0]) in visitados):
            visitados.append((indice[0], indice[1]))
            if valor_dist <= 0.1:
                if indice[0] not in del_idx_m:
                    del_idx_m.append(indice[0])
                if indice[1] not in del_idx_b:    
                    del_idx_b.append(indice[1])
    
    print("INICIO Patches a serem eliminados")            
    print (str(del_idx_b))
    print (str(del_idx_m))        
    print("FIM Patches a serem eliminados")            
    
    print("Total de atributos de cada base")            
    print ("Benignos: {0}".format(atribs_b.shape))
    print ("Malignos: {0}".format(atribs_m.shape))
    
    # Elimina os indices marcados
    novo_atribs_b = np.delete(atribs_b, del_idx_b, axis=0)
    novo_atribs_m = np.delete(atribs_m, del_idx_m, axis=0)
    
    print("Total de atributos de cada base APOS DELECAO")            
    print ("Benignos: {0}".format(novo_atribs_b.shape))
    print ("Malignos: {0}".format(novo_atribs_m.shape))
    
    # Atualiza o dicionario de imagens
    for idx in del_idx_m:
        dict_base.delPatch(idx_ptxs_m[idx])
        print("MALIGNOS: Deletando {0} na posicao {1}".format(idx_ptxs_m[idx], idx))
        
    for idx in del_idx_b:
        dict_base.delPatch(idx_ptxs_b[idx])
        print("BENIGNOS: Deletando {0} na posicao {1}".format(idx_ptxs_b[idx], idx))
        
    # Verifica se alguma imagem teve todos os patches deletados
    del_imgs = []
    for img in dict_base.imagens:
        if len(img["ids_patches"]) == 0:
            del_imgs.append(img)
            print("Lista de patches da imagem {0} esta vazia".format(img['arquivo']))
    
    #for img in del_imgs:
    #    dict_base.imagens.remove(img)
    
    # Acrescenta os labels
    novo_label_b = np.zeros((novo_atribs_b.shape[0],1))
    novo_label_m = np.ones((novo_atribs_m.shape[0],1))
    reduz_base_labels = np.concatenate((novo_label_b, novo_label_m), axis=0)
    
    #novo_atribs_b = np.concatenate((novo_label_b, novo_atribs_b), axis=1)
    #novo_atribs_m = np.concatenate((novo_label_m, novo_atribs_m), axis=1)
    
    reduz_base_atribs = np.concatenate((novo_atribs_b, novo_atribs_m), axis=0)
    '''    
    nova_base_atribs = BaseAtributos(DIR_BASE+'reduz-'+ARQ_BASE, tam_atribs=TAM_ATRIBS, usa_qid=True)
    nova_base_atribs.addBaseAtribs(reduz_base_atribs, list(reduz_base_labels), qids=[i for i in range(reduz_base_labels.shape[0])])
    nova_base_atribs.gravaArq()
    '''
    print("ATRIBUTOS - Qtde de linhas: {0}".format(reduz_base_atribs.shape[0]))
    print("LABELS - Qtde de linhas: {0}".format(reduz_base_labels.shape[0]))    
    
    dump_svmlight_file(reduz_base_atribs, reduz_base_labels.ravel(), DIR_BASE+'reduz-'+ARQ_BASE, query_id=[i for i in range(reduz_base_labels.shape[0])])
    pickle.dump( dict_base, open(DIR_BASE+'reduz-'+DICT_BASE, "wb" ) )
    
    
    
    
    
        
            