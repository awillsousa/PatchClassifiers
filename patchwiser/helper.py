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


cores = ['blue', 'green', 'red', 'grey',  'purple',
         'magenta', 'yellow', 'cyan', 'indigo', 
         'tomato', 'maroon', 'gold',  'crimson', 
         'teal', 'firebrick'] 

'''
Exibe o tempo passado a partir do inicio passado e retorna o tempo atual
em uma nova variável
'''    
def exibe_tempo(inicio, desc=""):
    fim = round(time() - inicio, 3)
    print ("Tempo total de execução ["+desc+"]: " +str(fim))
    
    return(time())     
      
'''
Retorna o resultado de um "ou-exclusivo para n variaveis" 
'''
def xor_n(*args):
    #return (sum(bool(a) for a in args) == 1)
    result = False
    for a in args:
        if a:
            if result:
                return False
            else:
                result = True
    return result
 
'''
Carrega um arquivo .csv em uma lista
'''
def load_csv(arquivo):
    with open(arquivo, 'r') as f:
        reader = csv.reader(f, quoting=csv.QUOTE_ALL)          
        lista = list(reader)
    
    return (lista)

'''      
Insere a informação do erro no log do programa e forca saida, encerrando o programa
'''
def loga_sai(erro):
    logging.info(erro)
    sys.exit(erro)    
    
'''
Verifica se uma opção passada existe na lista de argumentos do parser
'''  
def existe_opt (parser, dest):
   if any (opt.dest == dest and (opt._long_opts[0] in sys.argv[1:] or opt._short_opts[0] in sys.argv[1:]) for opt in parser._get_all_options()):
      return True
   return False     

'''
Gera o nome de todas as bases de treino e testes da base BreaKHis
'''
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
Realiza a plotagem simples de um grafico
'''
def plota_grafico2(dadosX, dadosY, arquivo="grafico.pdf", titulo="", tituloX="X", tituloY="Y", legendas=None,anota=False):
    # verifica se ha mais de um grafico para plotar    
    if isinstance(dadosX, list) and isinstance(dadosY, list):
       if not(hasattr(dadosX[0], "__iter__") or hasattr(dadosY[0], "__iter__")):
            dadosX = [dadosX]
            dadosY = [dadosY]
    else:
       loga_sai("Esperado lista!")

    print(str(dadosX))                
    print(str(dadosY))                
    plt.xlim(0.0, max([max(px) for px in dadosX]))            
    plt.ylim(0.0, max([max(py) for py in dadosY]))
    
    for i,(X,Y) in enumerate(zip(dadosX, dadosY)):
        # plota grafico de resultados        
        if legendas:
            legenda = legendas[i]
        else:
            legenda = ""
            
        plt.plot(X, Y, color=cores[i], label=legenda)
        
        # anota os pontos de classificacao
        if anota:
            for x,y in zip(dadosX,dadosY):       
                plt.plot([x,x],[0,y], color ='green', linewidth=.5, linestyle="--")
                plt.plot([x,0],[y,y], color ='green', linewidth=.5, linestyle="--")
                plt.scatter([x,],[y,], 50, color ='red')
        
    plt.ylabel(tituloY)
    plt.xlabel(tituloX)
    if (titulo == ""):
        titulo = tituloX + " vs " + tituloY
    plt.title(titulo)
    
    if legendas:
        #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        box = plt.get_position()
        plt.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=3)
    
    # salva arquivo    
    plt.savefig("./plots/plt-{0}.png".format(arquivo))
    plt.clf()


'''
Realiza a plotagem simples de um grafico
'''
def plota_grafico(dadosX, dadosY, arquivo="grafico.pdf", titulo="", tituloX="X", tituloY="Y", legendas=None,anota=False, plt0_1=False):
    # verifica se ha mais de um grafico para plotar    
    if isinstance(dadosX, list) and isinstance(dadosY, list):
       if not(hasattr(dadosX[0], "__iter__") or hasattr(dadosY[0], "__iter__")):
            dadosX = [dadosX]
            dadosY = [dadosY]
    else:
       loga_sai("Esperado lista!")

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
            
        ax.plot(X, Y, color=cores[i], label=legenda)
        
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
    plt.savefig("./plots/plt-{0}.png".format(arquivo))
    plt.clf()
    

'''
Exibe a visualizacao da projecao de uma base de n-atributos em um plano 2D utilizando
tSNE
'''
def visualiza_base(base, id_arquivo="data", texto=''):
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
    
    plt.savefig("plt-tSNE-{0}.png".format(id_arquivo))
    plt.close()

def visualiza_bhtsne(base, arq_plot, texto='',perp=33):    
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
    plt.savefig("./plots/plt-BHtSNE-{0}.png".format(arq_plot))
    plt.close() 


'''
Plota curvas roc
'''
def plot_rocs(rocs, labels, id_arquivo='AUC', texto='', titulo='Curvas ROC', lw=2):    
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
    plt.savefig("./plots/plt-ROC-{0}.png".format(id_arquivo))
    plt.close()

    
'''
Plota curva roc
'''
def plot_roc(fpr, tpr, roc_auc, label, id_arquivo='AUC', titulo='Curva ROC', color='maroon', lw=2):    

    label="AUC: {0} {1}".format(round(roc_auc,2),label)  
    plt.plot(fpr, tpr, lw=lw, color=color, label=label)            
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='')
    
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(titulo)
    plt.legend(loc="lower right")    
    plt.savefig("./plots/plt-ROC-{0}.png".format(id_arquivo))
    plt.close()

'''
Carrega uma base a partir de um arquivo .svm
'''
def carrega_base(arq_base, n_features=162, usa_qid=True):
    atribs = None 
    rotulos = None 
    qid = None            
    
    if usa_qid:
        atribs, rotulos, qid = load_svmlight_file(arq_base, n_features=n_features, query_id=usa_qid)
    else:
        atribs, rotulos = load_svmlight_file(arq_base, n_features=n_features)
        
    
    return (atribs, rotulos, qid)
