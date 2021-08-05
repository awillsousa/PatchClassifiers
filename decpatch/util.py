#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 17:17:09 2018

@author: willian

Funcoes utilitarias

"""
from random import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
from keras import backend as K

import fnmatch
import ConfVars
import mahotas as mh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle   
from matplotlib import colors as mcolors
plt.switch_backend('agg')
import os
from os import path
from cv2 import imread
import sys

# Lista todos os arquivos de imagens (png) que estejam abaixo de um diretório (em todas as profundidades)
def lista_imgs(dir_imagens, arquivo_labels=None):
    
    if arquivo_labels:
        with open(arquivo_labels, 'r') as f:
            arqs_imgs = [os.path.join(dir_imagens,linha).replace('\n', '') for linha in f.readlines()] 
    else:
        arqs_imgs = [os.path.join(raiz,f) for raiz,_,files in os.walk(dir_imagens) for f in files if fnmatch.fnmatch(f,'*.png')]
        
    print("Total de arquivos:{0}".format(len(arqs_imgs))) 
    shuffle(arqs_imgs)
    
    return (arqs_imgs) 


# Carrega uma imagem de devolve uma matriz Numpy
def carrega_imagem(arq_img):    
    img = mh.imread(arq_img)
    return (img)


# Recupera o label de um arquivo/imagem
def get_label(arquivo, base='cancer'):    
    if base == "especies":
        info_arquivo = arquivo.split('/')  
        if info_arquivo:
           classe = str(info_arquivo[-3]) 
           subclasse = str(info_arquivo[-2])    
        else:
           print("Problema ao recuperar o rotulo/classe da imagem.")

        return (ConfVars.ESPECIES_CLASSES[classe])
    elif base == "cancer":
        info_arquivo =  str(arquivo[arquivo.rfind("/")+1:]).split('_')        
        
        if info_arquivo:
            classe = info_arquivo[1]            
            #subclasse = info_arquivo[2].split('-')[0]
        else:
            print("Problema ao recuperar o rotulo/classe da imagem.")
            
        return (ConfVars.ROTULOS_CLASSES[classe])


def gera_patches(imagem, arquivo, rotulo, tamanho, desloca):
    '''
     Gera um conjunto de patches a partir da imagem passada
     Entrada:
     imagem - matriz contendo dados da imagem
     input_shape - formato do patch a extrair
     stride - deslocamento a cada extracao
     
     Saida:
         Uma lista de dicionarios com a seguinte configuracao:
         patches = {'posicao': (x,y), 'tamanho': (h,w), 'dados': <matrix com dados da imagem/patch>}    
    '''        
    patches = []
    altura,largura = imagem.shape[:2]      # duas primeiras dimensoes do tamanho da imagem
    #print("Altura: {0} Largura: {1}".format(altura, largura))
    margem_v = ((altura-tamanho) % desloca) // 2 
    margem_h = ((largura-tamanho) % desloca) // 2 
    #print("Margem V: {0} Margem H: {1}".format(margem_v, margem_h))                               
    i=0
    for linha in range(margem_v, altura-tamanho+1, desloca):            
        for coluna in range(margem_h, largura-tamanho+1, desloca):  
            #print("Patch: ({0}-{1},{2}-{3})".format(linha,linha+tamanho, coluna,coluna+tamanho)) 
            p = imagem[linha:linha+tamanho, coluna:coluna+tamanho]
            posicao = (coluna,linha)
            patches.append({'id':i, 'arquivo': arquivo, 'tam': (tamanho,tamanho), 'pos': posicao, 'patch': p, 'rotulo': rotulo, 'subpatches': []})  
            i+=1
            
    #print("Extraidos {0} patches".format(str(len(patches))))
    
    return (patches)

def prep_bases(arqs_imgs, subimg_size, subimg_delta, subptx_size=None, subptx_delta=None, gera_dic=True, base='cancer'):
    '''
    Entrada:
        arqs_imgs: lista de nomes dos arquivos de imagens (caminho completo)
        subimg_size: tamanho do patch (subimagem) a ser extraida
        subimg_delta: deslocamento da janela de extracao de subimagens
        subptx_size: tamanho do patch a ser extraido
        subptx_delta: tamanho do deslocamento da janela de extracao de patches
        
    Saida:
        imagens: dicionario de imagens com o seguinte formato: 
                    {'id': id_img, 'arquivo': arq_img, 'subimgs': subimgs}
                    'subimgs' contém um outro dicionario com o formato:
                        {'id':i, 'arquivo': arquivo, 'tam': (tamanho,tamanho), 'pos': posicao, 'patch': p, 'rotulo': rotulo, 'subpatches': []}
        X_1: conjunto de instancias (patches)
        y_1: conjunto de rotulos (labels) 
    '''
    X_1 = []    
    y_1 = []
    imagens = []        
    # Itera todas as imagens do conjunto de treino    
    for id_img,arq_img in enumerate(arqs_imgs):         
        print("Extraindo patches da imagem {0}".format(arq_img))        
        imagem = carrega_imagem(arq_img)    
        rotulo = get_label(arq_img, base)
        
        # Extrai subimagens
        # Para cada imagem somente sera extraido o patch mais discriminante,
        # por isso subdividi as imagens para termos mais patches selecionados por imagem
        subimgs = gera_patches(imagem, arq_img, rotulo, tamanho=subimg_size, desloca=subimg_delta)
        del imagem        
        if subptx_size:
            # Para cada uma das subimagens gera um conjunto de patches que sera utilizado para o treinamento da CNN
            for subimg in subimgs:            
                patches = gera_patches(subimg['patch'], subimg['arquivo'], subimg['rotulo'], tamanho=subptx_size, desloca=subptx_delta)            
                for p in patches:
                    p['pos'] = p['pos'] + subimg['pos']
                    X_1.append(p['patch'])
                    y_1.append(p['rotulo'])
                subimg['subpatches'] = patches
                
                subimg['patch'] = None # nao precisa guardar a matrix do subpatch
        if gera_dic:        
            imagens.append({'id': id_img, 'arquivo': arq_img, 'subimgs': subimgs})    

    return (X_1, y_1, imagens)

def prep_bases2(arqs_imgs, subptx_size=None, subptx_delta=None, gera_dic=True, base='cancer'):
    '''
    Entrada:
        arqs_imgs: lista de nomes dos arquivos de imagens (caminho completo)        
        subptx_size: tamanho do patch a ser extraido
        subptx_delta: tamanho do deslocamento da janela de extracao de patches
        
    Saida:
        imagens: dicionario de imagens com o seguinte formato: 
                    {'id': id_img, 'arquivo': arq_img, 'subimgs': subimgs}
                    'subimgs' contém um outro dicionario com o formato:
                        {'id':i, 'arquivo': arquivo, 'tam': (tamanho,tamanho), 'pos': posicao, 'patch': p, 'rotulo': rotulo, 'subpatches': []}
        X_1: conjunto de instancias (patches)
        y_1: conjunto de rotulos (labels) 
    '''
    X_1 = [] 
    y_1 = []
    imagens = [] 
    # Itera todas as imagens do conjunto de treino    
    for id_img,arq_img in enumerate(arqs_imgs):         
        print("Extraindo patches da imagem {0}".format(arq_img))        
        imagem = carrega_imagem(arq_img)    
        rotulo = get_label(arq_img, base)
        
        # Extrai subimagens
        # Para cada imagem somente sera extraido o patch mais discriminante,
        # por isso subdividi as imagens para termos mais patches selecionados por imagem
        patches = gera_patches(imagem, arq_img, rotulo, tamanho=subptx_size, desloca=subptx_delta)
        for p in patches: 
            X_1.append(p['patch'])
            y_1.append(p['rotulo'])
            
        if gera_dic:        
            imagens.append({'id': id_img, 'arquivo': arq_img, 'patches': patches}) 

    return (X_1, y_1, imagens)


# Cria o modelo da Fase 01 seguindo o padrão da CNN utilizada em CIFAR
def model_fase01_cifar(input_shape, num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    #model.add(Activation('softmax'))
    model.add(Activation('sigmoid'))
    
    return (model)


# Cria o modelo da Fase 01
def model_fase01(input_shape, num_classes, arquivo_pesos=None):    
    model = Sequential()
    model.add(Conv2D(20, (9, 9), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(40, (9, 9)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))    

    model.add(Flatten())
    model.add(Dense(600))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))
    model.add(Dense(num_classes))    
    model.add(Activation('softmax'))
    
    # Se já existe um conjunto de pesos calculados (um modelo treinado)
    # carrega os pesos no modelo e usa
    
    if arquivo_pesos: 
        model.load_weights(arquivo_pesos)
    
    return model


# Plota o historico de aprendizado de um modelo 
def plot_history(cnn):    
    dirtime = datetime.now().strftime('%Y%m%d-%H%M%S')
    PLOT_DIR = "./plots/"+dirtime
    if not os.path.isdir(PLOT_DIR):
        os.makedirs(PLOT_DIR)
    # Plots for training and testing process: loss and accuracy 
    plt.figure(0)
    plt.plot(cnn.history['acc'],'r')
    plt.plot(cnn.history['val_acc'],'g')
    plt.xticks(np.arange(0, 101, 10.0))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num. Epochs")
    plt.ylabel("Acuracia")
    plt.title("Treino Acuracia x Validacao Acuracia")
    plt.legend(['treino','validacao'])
    plt.savefig(PLOT_DIR+"/fase01-acc.png")
    
    plt.figure(1)
    plt.plot(cnn.history['loss'],'r')
    plt.plot(cnn.history['val_loss'],'g')
    plt.xticks(np.arange(0, 101, 10.0))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num. Epochs")
    plt.ylabel("Custo")
    plt.title("Treino Custo x Validacao Custo")
    plt.legend(['treino','validacao'])
    plt.savefig(PLOT_DIR+"/fase01-loss.png")
    #plt.show()
    
# Encontra o ultimo arquivo de checkpoint
def last_ckpt(dir):
    fl = os.listdir(dir)
    fl = [x for x in fl if x.endswith(".hdf5")]
    cf = ""
    if len(fl) > 0:
        accs = [float(x.split("-")[3][0:-5]) for x in fl]
        m = max(accs)
        iaccs = [i for i, j in enumerate(accs) if j == m]
        fl = [fl[x] for x in iaccs]
        epochs = [int(x.split("-")[2]) for x in fl]
        cf = fl[epochs.index(max(epochs))]
        cf = os.path.join(dir,cf)
  
    return(cf)   


# Plota a matrix de confusao de um conjunto de teste
def conf_matrix(y_test, y_pred):    
    dirtime = datetime.now().strftime('%Y%m%d-%H%M%S')
    PLOT_DIR = "./plots/"+dirtime
    
    for ix in range(10):
        print(ix, confusion_matrix(np.argmax(y_test,axis=1),y_pred)[ix].sum())
    cm = confusion_matrix(np.argmax(y_test,axis=1),y_pred)
    print(cm)
     
    # Visualizing of confusion matrix
    import seaborn as sn
    import pandas  as pd     
     
    df_cm = pd.DataFrame(cm, range(10), range(10))
    plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4) #for label size
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 12}) # font size    
    plt.savefig(PLOT_DIR+"/confusion-matrix.png")
    plt.show()
    
# Plota os dados do treinamento do modelo
def plot_model_history(model_history, file='fase01-loss.png'):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.savefig("plots/"+file)
    #plt.show()


# Avalia a acuracia de um modelo
def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class) 
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)

# Devolve o valor de sensitividade de um conjunto
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

# Devolve o valor de especificidade de um conjunto 
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

# Marca uma lista de patches na sua imagem de origem
def marca_patches(arq_img, patches, legenda="X", tam_patch=64, rgb=True, saida="plots/ptxsmarcados.png",cores_rotulos=None):        
    # Se nao for passada uma lista de cores, cria uma
    if cores_rotulos is None:
       cores = list(mcolors.CSS4_COLORS.keys())
       shuffle(cores)
       # Dicionario de cores para cada classe    
       cores_rotulos = {}
       for p in patches:
           if p['rotulo'] not in cor_classe:
              cor_classe[p['rotulo']] = cores[0]
              del cores[0]

    if path.isfile(arq_img):        
       img = imread(arq_img)        
    else:
        sys.exit("Arquivo de imagem incorreto - {0}".format(arq_img))
    
    fig_size = plt.rcParams["figure.figsize"]    
    fig_size[0] = 12
    fig_size[1] = 9
    plt.rcParams["figure.figsize"] = fig_size
        
    # Cria figura
    fig,ax = plt.subplots(1)                
    ax.imshow(img)    
    for p in patches:
        ax.add_patch(Rectangle(p['pos'], *p['tam'],alpha=0.5,lw=1,
                               facecolor=cores_rotulos[p['rotulo']], 
                               edgecolor="black")) 
    #plt.show()
    plt.savefig("plots/{}".format(saida))
    plt.clf() 
    plt.close(fig)

def gera_cores_classes(dict_rotulos):
    
    # Lista de cores com 148 opcoes
    cores = list(mcolors.CSS4_COLORS.keys())
    r = [c for c in cores if "red" in c]    
    g = [c for c in cores if "green" in c]
    b = [c for c in cores if "blue" in c]
    outros = [c for c in cores if "red" not in c \
                               and "green" not in c \
                               and "blue" not in c]
    cores = r+g+b
    shuffle(cores)
    cores += outros
    
    # Dicionario de cores para cada classe    
    cor_classes = {}
    for i in dict_rotulos.values():
        cor_classes[i] = cores[i]
    

    return (cor_classes)
