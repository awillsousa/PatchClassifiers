#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 03:02:58 2017

@author: willian

Avalia a classificação de imagens da base de cancer, baseado nas probabilidades
geradas através de um SVM
As imagens serão classificadas considerando os votos de todos os patches da imagem. 
Os patches serão escolhidos a partir de um determinado limiar da probabilidade da classe
estar correta. 
"""

import helper
import logging
import extrator as ex
import csv
import numpy as np
from sklearn.metrics import accuracy_score
from datetime import datetime
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


'''
Organiza um dicionario de imagens a partir de uma lista de strings passada
'''
def dicionario_imgs(lista_ppi):
    lista = []     
    for l in lista_ppi:
        d = {
              'ppi':int(l[1]),          # patches por imagem
              'total':int(l[2]),        # total de patches 
              'descartados':int(l[3]),  # patches descartados
              'classe':l[4],            # classe da imagem
              'arquivo': l[5],           # caminho do arquivo da imagem
              'rotulo' : '',
              'predicao' : ''              
            }
        lista.append(d) 
    return (lista)

def classifica_img(imagem, classe, preds_img, probs_img, limiar):
    #logging.info("Classificacao imagem " + imagem)
    
    # recupera o rotulo real da imagem    
    rotulo_real = ex.CLASSES[classe]
    
    # utiliza apenas as prediçoes dos patches cuja probabilidade esteja
    # acima do limiar       
    total = probs_img.shape[0]    
    preds_img = preds_img[np.where(probs_img > limiar)]
    validos = preds_img.shape[0]
    
    
    descarte = total - validos
    errados = len([x for x in preds_img if x != rotulo_real])
    
    '''
    print("\nErrados {0}".format(errados))    
    
    print("Validos {0}".format(validos))
    print("Descartados {0}".format(descarte))
    '''
        
    if (validos > 0):
        conta = np.bincount(preds_img)                
        rotulo_pred = np.argmax(conta)        
    else:
        rotulo_pred = -1
        
    return (rotulo_real, rotulo_pred, descarte, errados)
   


def processa_proba(arq_pred, arq_ppi, limiar):
    # carrega arquivo de patches por imagem da base de teste            
    imagens = dicionario_imgs(helper.load_csv(arq_ppi))
    logging.info("Carregado arquivo de quantidade de patches por imagem: " + arq_ppi)
    
    # carrega arquivo de predicoes de patches
    with open(arq_pred, 'r') as f:
        next(f)
        reader = csv.reader(f, quoting=csv.QUOTE_ALL,delimiter=" ")
        probs_patches = list(reader)
        
    preds = np.array([int(x[0]) for x in probs_patches])
    probs = np.array([max(float(x[1]),float(x[2])) for x in probs_patches])
    
    # realiza classificacao das imagens por voto, baseado no arquivo de predicoes
    # gerados por um SVM
    
    idx1 = 0    # posicao inicial dos atributos da imagem
    idx2 = 0    # posicao final dos atributos da imagem
    if not (imagens):
       print("Algo de errado com a lista de imagens!")
    print("Tamanho das lista de imagens: {0}".format(len(imagens)))
    num_ppi = imagens[0]['total']
    total_desc = 0      # total de patches descartados
    total_erro = 0
    imgs_desc = 0      # total de imagens descartadas (todos os patches descartados)    
    imgs_erro = 0
    desc_porimg = [] 
    y_test = [] 
    y_pred = []
    
    # Carrega os atributos de acordo com as informações do arquivos de patches por imagem (.ppi)
    for imagem in imagens:
        idx2 = imagem['ppi']            
        if idx2 > 0:
            idx2 += idx1  # limite superior da fatia                
            preds_img = preds[idx1:idx2]
            probs_img = probs[idx1:idx2]
            tst,pred,desc,erro = classifica_img(imagem['arquivo'], imagem['classe'], preds_img, probs_img, limiar)
            total_desc += desc
            total_erro += erro
            desc_porimg.append(desc)
            
            if pred == -1:
                imgs_desc += 1            
            else:
                y_test.append(tst)
                y_pred.append(pred)
                
                if tst != pred:
                   imgs_erro += 1                
            idx1 = idx2
            
    # Resultados
    total_imgs = len(imagens)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    total_patches = total_imgs*num_ppi
    
    r = {'matriz' : cnf_matrix,
         'descarte' : total_desc/total_patches,
         'img_desc' : imgs_desc/total_imgs,
         'acc' : accuracy_score(y_test, y_pred),         
         'erro' :  total_erro/total_patches,
         'img_erro' : imgs_erro/total_imgs}   
    return (r)

def plota(eixoX, eixosY, labelX, labelY, titulo, min_limiar=0.0):
    cores = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']   
    plt.clf()
    for i,eixo in enumerate(eixosY):
        plt.plot(eixoX, eixo, lw=1, color=cores[i], label='fold'+str(i+1))
        
    plt.xlabel(labelX)
    plt.ylabel(labelY)
    plt.title('Magnitude ' + mag)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)    
    plt.show()
    
   
    
#########################################################################################

## Cria a entrada de log do programa
#idarq=datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
#logging.basicConfig(filename='probas-'+idarq+'.log', format='%(message)s', level=logging.INFO)  
  
'''
# Configura para o log ser exibido na tela
console = logging.StreamHandler()
console.setLevel(logging.INFO)            
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)   
logging.getLogger('').addHandler(console)
'''

##logging.info("INICIO DO PROGRAMA")   

folds = ['fold1','fold2','fold3','fold4','fold5']
mags = ['40X', '100X', '200X', '400X']

base="/folds/execs/bases/cancer/"
#base="Y:/bases/execs/"
'''
arqs_ppi = ["/folds/execs/bases/cancer/fold5/f5_ts_400X_pftas_ptx64x64.ppi",
            "/folds/execs/bases/cancer/fold5/f4_ts_400X_pftas_ptx64x64.ppi",
            "/folds/execs/bases/cancer/fold5/f3_ts_400X_pftas_ptx64x64.ppi"
            "/folds/execs/bases/cancer/fold5/f2_ts_400X_pftas_ptx64x64.ppi"
            "/folds/execs/bases/cancer/fold5/f1_ts_400X_pftas_ptx64x64.ppi"]
arqs_pred = ["/folds/execs/bases/cancer/fold5/f5_ts_400X_pftas_ptx64x64.pred",
             "/folds/execs/bases/cancer/fold5/f4_ts_400X_pftas_ptx64x64.pred",
             "/folds/execs/bases/cancer/fold5/f3_ts_400X_pftas_ptx64x64.pred",
             "/folds/execs/bases/cancer/fold5/f2_ts_400X_pftas_ptx64x64.pred",
             "/folds/execs/bases/cancer/fold5/f1_ts_400X_pftas_ptx64x64.pred"]
'''
cores = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']   
arqs_ppi = []
arqs_pred = []
for mag in mags:        
    plots_acc = []
    plots_descarte = []
    plots_imgdesc = []
    plots_imgerro = []
    plots_erro = []
    for i,fold in enumerate(folds):    
        print("Magnitude {0} - Fold {1}".format(mag, fold))
        arq_ppi = base+fold+"/test/"+mag+"/base_pftas_patches64x64.ppi"
        arq_pred = base+fold+"/test/"+mag+"/base_pftas_patches64x64.pred"

        acc = []
        descarte = []
        imgdescarte = []
        imgerro = []
        limiares = []
        erros = []        
        min_limiar = 0.00
        max_limiar = 1.00
        incr = 0.01
        limiar = min_limiar           
        #for i in range(800,999,1):
        while limiar >= min_limiar and limiar <= max_limiar:
            print("Limiar {0}".format(limiar))
            limiar += incr
            r = processa_proba(arq_pred, arq_ppi, limiar)
                
            acc.append(r['acc'])
            limiares.append(limiar)
            descarte.append(r['descarte'])
            imgdescarte.append(r['img_desc'])
            erros.append(r['erro'])
            imgerro.append(r['img_erro'])
            #print("Limiar {0:.3f} - Acc {1:.4f}".format(limiar, r['acc']))
        plots_acc.append(acc)
        plots_descarte.append(descarte)
        plots_imgdesc.append(imgdescarte)
        plots_imgerro.append(imgerro)
        plots_erro.append(erros)
        
    plt.clf()
    for i,(descartes,erros) in enumerate(zip(plots_imgdesc,plots_imgerro)):
        plt.plot(descartes,erros, lw=1, color=cores[i], label='fold'+str(i+1))
    plt.xlabel('Taxa Rejeição(%)')
    plt.ylabel('Taxa Erro(%)')
    plt.title('Magnitude ' + mag + " (Imagens)")
    plt.ylim([0.0, 1.01])
    plt.xlim([0.0, 1.01])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)    
    plt.show()
    
    plt.clf()
    for i,(descartes,erros) in enumerate(zip(plots_descarte,plots_erro)):
        plt.plot(descartes,erros, lw=1, color=cores[i], label='fold'+str(i+1))
    plt.xlabel('Taxa Rejeição(%)')
    plt.ylabel('Taxa Erro(%)')
    plt.title('Magnitude ' + mag + " (Patches)")
    plt.ylim([0.0, 1.01])
    plt.xlim([0.0, 1.01])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)    
    plt.show()
    
    
    plt.clf()
    for i,(descartes,erros) in enumerate(zip(plots_descarte,plots_imgerro)):
        plt.plot(descartes,erros, lw=1, color=cores[i], label='fold'+str(i+1))
    plt.xlabel('Taxa Rejeição Patches(%)')
    plt.ylabel('Taxa Erro Imagens(%)')
    plt.title('Magnitude ' + mag + " (Imagens)")
    plt.ylim([0.0, 1.01])
    plt.xlim([0.0, 1.01])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)    
    plt.show()
    
    #plota(limiares, plots_acc, 'Probabilidade', 'Acc', 'Magnitude ' + mag, min_limiar=0.0)
    #plota(limiares, plots_descarte, 'Probabilidade', 'Descarte', 'Magnitude ' + mag, min_limiar=0.0)
    
    '''
    # Plotagem das curvas de acc, descarte e descarte de imagens (separados) para todos os folds 
    # e para cada magnitude, pelo limiar de descarte baseado em probabilidade
    plota(limiares, plots_acc, 'Probabilidade', 'Acc', 'Magnitude ' + mag, min_limiar=0.0)
    plota(limiares, plots_descarte, 'Probabilidade', 'Descarte', 'Magnitude ' + mag, min_limiar=0.0)
    plota(limiares, plots_imgdesc, 'Probabilidade', 'Descarte (Imagens)', 'Magnitude ' + mag, min_limiar=0.0)
    
    # Plotagem de acc, descarte de patches e descarte de imagens pelo limiar
    # de descarte baseado em probabilidade
    for i, (acc,desc,img) in enumerate(zip(plots_acc,plots_descarte,plots_imgdesc)):
        plt.clf()        
        plt.plot(limiares, acc, lw=1, color=cores[0], label='Acc')
        plt.plot(limiares, desc, lw=1, color=cores[2], label='Desc Patches')
        plt.plot(limiares, img, lw=1, color=cores[4], label='Desc Imagens')
        plt.xlabel('Probabilidade')
        #plt.ylabel('Acc')
        plt.title('Fold {0} - Magnitude {1}'.format(i+1,mag))
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.05])
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        #plt.legend(loc="lower left")
        plt.show()    
    
    
    
    plt.clf()
    for i,limiares in enumerate(plots_descarte):
        plt.plot(limiares, descarte, lw=1, color=cores[i], label='fold'+str(i+1))
    plt.xlabel('Probabilidade')
    plt.ylabel('Descarte')
    plt.title('Magnitude ' + mag)
    plt.ylim([0.75, 1.05])
    plt.xlim([min_limiar, 1.0])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.legend(loc="lower left")
    plt.show()
    '''
#logging.info("ENCERRAMENTO DO PROGRAMA")  
