# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 02:17:38 2016

@author: willian

Funções utilitárias e de uso comum
"""

import logging
import sys
import csv
import numpy as np
import bhtsne
from time import time
from sklearn.decomposition import PCA
from sklearn.datasets import load_svmlight_file
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
from numpy import float32
import matplotlib.pyplot as plt



class Display():
    
    cores = ['blue', 'green', 'red', 'grey',  'purple',
         'magenta', 'yellow', 'cyan', 'indigo', 
         'tomato', 'maroon', 'gold',  'crimson', 
         'teal', 'firebrick'] 

    '''
    Gera o nome de todas as bases de treino e testes da base BreaKHis
    '''
    @staticmethod
    def todas_breakhis():
        folds = ['fold1','fold2','fold3','fold4','fold5']
        mags = ['40X', '100X', '200X', '400X']    
        caminho_bases ="/home/awsousa/bases/execs/cancer/"    
        bases_mag = {m:{f: {'tr': {'pftas': "", 'glcm': ""}, 'ts':  {'pftas': "", 'glcm': ""}} for f in folds} for m in mags}
        
        for mag, folds in bases_mag.items():
            for fold, base in folds.items():            
                base['tr']['pftas'] = caminho_bases+fold+"/train/"+mag+"/base_pftas_ptx64x64.svm"
                base['ts']['pftas'] = caminho_bases+fold+"/test/"+mag+"/base_pftas_ptx64x64.svm"
                base['tr']['glcm'] = caminho_bases+fold+"/train/"+mag+"/base_pftas_ptx64x64.glcm"
                base['ts']['glcm'] = caminho_bases+fold+"/test/"+mag+"/base_pftas_ptx64x64.glcm"
                
        return (bases_mag)

    '''
    Gera o nome de todas as bases de treino e testes da base BreaKHis
    '''
    @staticmethod
    def todas(q):
        folds = ['fold1','fold2','fold3','fold4','fold5']
        if q == 40:
            mags = ['40X']
        elif q == 100:
            mags = ['100X']    
        elif q == 200:
            mags = ['200X']    
        elif q == 400:
            mags = ['400X']    
        else:
            caminho_bases ="/home/awsousa/bases/execs/cancer/"    
            bases_mag = {"treino":{"unica": {'tr': {'pftas': "", 'glcm': ""}, 'ts':  {'pftas': "", 'glcm': ""}}}}
            for mag, folds in bases_mag.items():
                for fold, base in folds.items():
                    if q == 0:
                        base['tr']['pftas'] = caminho_bases+"min_treino/base_pftas_ptx64x64.svm"
                        base['ts']['pftas'] = caminho_bases+"min_teste/base_pftas_ptx64x64.svm"
                        base['tr']['glcm'] = caminho_bases+"min_treino/base_pftas_ptx64x64.glcm"
                        base['ts']['glcm'] = caminho_bases+"min_teste/base_pftas_ptx64x64.glcm"
                    elif q == 1:
                        base['tr']['pftas'] = caminho_bases+"1px_treino/base_pftas_ptx64x64.svm"
                        base['ts']['pftas'] = caminho_bases+"1px_teste/base_pftas_ptx64x64.svm"
                        base['tr']['glcm'] = caminho_bases+"1px_treino/base_pftas_ptx64x64.glcm"
                        base['ts']['glcm'] = caminho_bases+"1px_teste/base_pftas_ptx64x64.glcm"
            
            return (bases_mag)

        caminho_bases ="/home/awsousa/bases/execs/cancer/"    
        bases_mag = {m:{f: {'tr': {'pftas': "", 'glcm': ""}, 'ts':  {'pftas': "", 'glcm': ""}} for f in folds} for m in mags}
        
        for mag, folds in bases_mag.items():
            for fold, base in folds.items():            
                base['tr']['pftas'] = caminho_bases+fold+"/train/"+mag+"/base_pftas_ptx64x64.svm"
                base['ts']['pftas'] = caminho_bases+fold+"/test/"+mag+"/base_pftas_ptx64x64.svm"
                base['tr']['glcm'] = caminho_bases+fold+"/train/"+mag+"/base_pftas_ptx64x64.glcm"
                base['ts']['glcm'] = caminho_bases+fold+"/test/"+mag+"/base_pftas_ptx64x64.glcm"
                
        return (bases_mag)

    '''
    Realiza a plotagem de um grafico ou de um conjunto de graficos.
    Os dados de cada eixo devem ser passados como uma lista ou uma lista de listas
    '''
    @staticmethod
    def plota_grafico(dadosX, dadosY, diretorio="./plots/", arquivo="plt", titulo="", tituloX="X", tituloY="Y", legendas=None,anota=False, plt0_1=False):
        # verifica se ha mais de um grafico para plotar    
        if isinstance(dadosX, list) and isinstance(dadosY, list):
           if not(hasattr(dadosX[0], "__iter__") or hasattr(dadosY[0], "__iter__")):
                dadosX = [dadosX]
                dadosY = [dadosY]
        else:
           logging.info("Esperado lista!")
    
        fig = plt.figure()
        ax = plt.subplot(111)
    
        ax.set_xlim(0.0, max([max(px) for px in dadosX]))            
        
        if plt0_1:
            ax.set_xlim(0.0, 1.0)            
        else:
            ax.set_ylim(0.0, max([max(py) for py in dadosY]))
            
        ax.grid(True)
            
        for i,(X,Y) in enumerate(zip(dadosX, dadosY)):
            # plota grafico de resultados        
            if legendas:
                legenda = legendas[i]
            else:
                legenda = ""
                
            ax.plot(X, Y, color=Display.cores[i], label=legenda)
            
            # anota os pontos de classificacao
            if anota:
                for x,y in zip(dadosX,dadosY):       
                    ax.plot([x,x],[0,y], color ='green', linewidth=.5, linestyle="--")
                    ax.plot([x,0],[y,y], color ='green', linewidth=.5, linestyle="--")
                    ax.scatter([x,],[y,], 50, color ='red')
            
        ax.set_ylabel(tituloY, fontsize=9)
        ax.set_xlabel(tituloX, fontsize=9)
        if (titulo == ""):
            titulo = tituloX + " vs " + tituloY
        ax.set_title(titulo)
        
        # Linhas de grade
        gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        for line in gridlines:
            line.set_linestyle('-.')
        
        if legendas:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
            
            # Put a legend below current axis
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11),
              fancybox=True, shadow=True, ncol=3)
        
        # salva arquivo    
        plt.savefig("{0}GRAPH-{1}.png".format(diretorio, arquivo))
        plt.clf()


    '''
    Exibe a visualizacao da projecao de uma base de n-atributos em um plano 2D utilizando
    tSNE. Esse modo utiliza apenas algumas amostras da base para a projecao.
    Pode ser utilizado para uma prototipação mais rápida.
    '''
    @staticmethod
    def visualiza_base(base, diretorio="./plots/", arquivo="base", texto=''):
        size = 50
        N = base["labels"].shape[0]/5#size * size
        
        if hasattr(base["data"],"todense"):
            data, target = shuffle(base['data'].todense(), base['labels'], random_state=777, n_samples=int(N))
        else:
            data, target = shuffle(base['data'], base['labels'], random_state=777, n_samples=int(N))
            
        data_100 = PCA(n_components=100).fit_transform(data.astype(float32) / 255)
        embeddings = TSNE(init="pca", random_state=777, verbose=0, perplexity=33.0, early_exaggeration=4.0, learning_rate=500.0, n_iter=1000).fit_transform(data_100)
        embeddings -= embeddings.min(axis=0)
        embeddings /= embeddings.max(axis=0)
        
        plt.rcParams["figure.figsize"] = (17, 9)
        plt.scatter(embeddings[:, 0], embeddings[:, 1], c=target)
        my_colorbar = plt.colorbar(fraction=0.05, pad = 0.0125)
        plt.xticks([])
        plt.yticks([])
        
        if texto != '':
           plt.text(1.15, 0, texto)
        
        # salva arquivo    
        plt.savefig("{0}bhtsne-{1}.png".format(diretorio, arquivo))
        plt.clf()

    '''
    Exibe a visualizacao da projecao de uma base de n-atributos em um plano 2D utilizando
    tSNE. Utiliza uma quantidade maior de instâncias das base. Sendo mais demorado para
    executar.
    '''
    @staticmethod    
    def visualiza_bhtsne(base, diretorio="./plots/", arquivo="base", texto='',perp=33):    
        # Carrega arquivo
        #atrib_ts, rotulos_ts = load_svmlight_file(base, dtype=np.float32, n_features=162)
        data = base['data']
        labels = base['labels']
        
        if hasattr(data,"toarray"):
            embeddings = bhtsne.run_bh_tsne(data.toarray(), initial_dims=data.shape[1],perplexity=perp)
        elif hasattr(data,"todense"):
            embeddings = bhtsne.run_bh_tsne(data.todense(), initial_dims=data.shape[1],perplexity=perp)
        else:
            embeddings = bhtsne.run_bh_tsne(np.asarray(data), initial_dims=data.shape[1],perplexity=perp)
        
        plt.rcParams['figure.figsize'] = (17, 9)
        plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels)
        my_colorbar = plt.colorbar(fraction=0.05, pad = 0.0125)
        plt.xticks([])
        plt.yticks([])    
        if texto != '':
           plt.text(1.15, 0, texto)    
        
        # salva arquivo    
        plt.savefig("{0}BHTSNE-{1}.png".format(diretorio, arquivo))
        plt.clf()


    '''
    Plota um conjunto de curvas roc
    '''
    @staticmethod
    def plot_rocs(rocs, labels, diretorio="./plots/", arquivo="AUC", texto='', titulo='Curvas ROC', lw=2):    
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
                
        for roc,color,label in zip(rocs, colors[:len(rocs)+1], labels):
            fpr = roc[0]
            tpr = roc[1]
            roc_auc = roc[2]    
            plt.plot(fpr, tpr, lw=lw, color=color, label=label+' AUC: %0.2f' % (roc_auc))            
            
        plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='')    
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(titulo)
        plt.legend(loc="lower right")
        if texto != '':
           plt.text(0, -0.5, texto)
        
        # salva arquivo    
        plt.savefig("{0}ROC-{1}.png".format(diretorio, arquivo))
        plt.clf()

        
    '''
    Plota curva roc
    '''
    @staticmethod
    def plot_roc(fpr, tpr, roc_auc, label, diretorio="./plots/", arquivo="AUC", titulo='Curva ROC', color='maroon', lw=2):    
    
        label="AUC: {0} {1}".format(round(roc_auc,2),label)  
        plt.plot(fpr, tpr, lw=lw, color=color, label=label)            
        plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='')
        
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(titulo)
        plt.legend(loc="lower right")    
        
        # salva arquivo    
        plt.savefig("{0}ROC-{1}.png".format(diretorio, arquivo))
        plt.clf()

