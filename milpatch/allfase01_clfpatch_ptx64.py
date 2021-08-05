'''Implementacao do metodo de selecao de patches por relevancia definido em 
Bodypart recognition using multi-stage deep learning.

Foi utilizado keras para simplificar a implementacao

Procedimentos a serem executados:
    
- Para cada imagem dividi-la em patches de tamanho 64x64
- Treinar a CNN utilizando uma funcao de custo multi-instancias baseadas nos patches
- Extrair o conjunto de patches relevantes (Am) e não-relevantes (Bm)
- Atribuir os labels corretos aos patches de Am e atribuir um novo label aos patches de Bm
- Modificar a CNN pre-treinada, inserindo a nova classe (de patches irrelevantes)
- Faz boosting na CNN pre-treinada utilizando (Am U Bm) como entrada e utiliza uma funcao de 
  custo baseada nas imagens para obter os pesos finais da rede
- Classifica as imagens utilizando as predicoes das classes relevantes, ignorando as predicoes
  dos patches classificados como irrelevantes.
'''

from sklearn.metrics import log_loss, confusion_matrix, classification_report,roc_auc_score
from sklearn.metrics import roc_curve, auc
from scipy import interp
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard
from keras import backend as K
from keras import optimizers
from datetime import time

import fnmatch
import argparse
import numpy as np
import mahotas as mh
import ConfVars
from milpatch import sensitivity, specificity
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os

FLAGS = None
NUM_CLASSES = 2
# Dimensao das imagens
SUBIMG_SIZE = 90
IMG_WIDTH, IMG_HEIGHT = 460, 700
PTX_WIDTH = PTX_HEIGHT = PTX_SIZE = 64
EPOCHS = 100
BATCH_SIZE = 25
_EPSILON = K.epsilon()
OPTIMIZER='adam'

if K.image_data_format() == 'channels_first':
    INPUT_SHAPE = (3, PTX_WIDTH, PTX_HEIGHT)
else:
    INPUT_SHAPE = (PTX_WIDTH, PTX_HEIGHT, 3)

def get_loss():
    def loss(y_true, y_pred):               
            y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)         
            maximos = K.max(y_pred, axis=1)            
            l = -K.sum(K.log(maximos), axis=-1)
            return (l)
    return loss

def create_model(arquivo_pesos=None):
    model = Sequential()
    model.add(Conv2D(20, (9, 9), input_shape=INPUT_SHAPE))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(40, (9, 9)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(600))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))

    # Se já existe um conjunto de pesos calculados (um modelo treinado)
    # carrega os pesos no modelo e usa

    if arquivo_pesos:
        model.load_weights(arquivo_pesos)

    model.compile(loss='binary_crossentropy',
                  optimizer=OPTIMIZER,
                  metrics=['accuracy'])
                  #metrics=['accuracy', sensitivity, specificity])                      

    return model

def carrega_imagem(arq_img):    
    img = mh.imread(arq_img)
    return (img)

def get_label(arquivo):    
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
            patches.append({'id':i, 'arquivo': arquivo, 'tam': (tamanho,tamanho), 'pos': posicao, 'array': p, 'rotulo': rotulo })  
            i+=1
            
    print("Extraidos {0} patches".format(str(len(patches))))
    
    return (patches)

def main():
    
    FOLDS_MODEL = [(1,'/home/willian/bases/cancer/fold1/test/400X/','fold1/pesos-fase01.h5'),
                   (2,'/home/willian/bases/cancer/fold2/test/400X/','fold2/pesos-fase01.h5'),
                   (3,'/home/willian/bases/cancer/fold3/test/400X/','fold3/novo-fase01.h5'),
                   (4,'/home/willian/bases/cancer/fold4/test/400X/','fold4/novo-fase01.h5'),
                   (5,'/home/willian/bases/cancer/fold5/test/400X/','fold5/novo-fase01.h5')]

    # listas auxiliares para plotagem das curvas ROC
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for num_fold,image_dir,pesos_fase01 in FOLDS_MODEL:
        print("Fold {}".format(num_fold))
        #Cria o modelo a ser treinado  
        arquivo_pesos = os.path.join('./models/'+pesos_fase01)
    
        if os.path.isfile(arquivo_pesos):    
            model = create_model(arquivo_pesos)
        else:    
            print("Modelo não encontrado!")
            os.sys.exit(1)
    
        # Gera uma lista das imagens a classificar    
        arqs_imgs = [os.path.join(image_dir,f) for f in os.listdir(image_dir) if fnmatch.fnmatch(f,'*.png')]
        arqs_imgs += [os.path.join(image_dir,f) for f in os.listdir(image_dir) if fnmatch.fnmatch(f,'*.jpg')]
    
        # Gera patches a partir das imagens originais. 
        # A classificacao das imagens sera feita em nivel de patch
        imagens = []
        # Itera todas as imagens do conjunto de treino
        for id_img,arq_img in enumerate(arqs_imgs):
            X_1 = []
            y_1 = []
            #print("Extraindo patches da imagem {0}".format(arq_img))        
            imagem = carrega_imagem(arq_img)        
            rotulo = get_label(arq_img)
        
            # Extrai subimagens de tamanho 90x90 com deslocamento de 60
            # Como as imagens são 800x600 e para imagem somente sera extraido o patch mais discriminante,
            # subdividi as imagens para termos mais patches selecionados por imagem
            patches = gera_patches(imagem, arq_img, rotulo, tamanho=PTX_SIZE, desloca=PTX_SIZE//2)
        
            for p in patches:
                X_1.append(p['array'])
                y_1.append(p['rotulo'])
        
            X = np.asarray(X_1, dtype='float32')
            y = np.asarray(y_1, dtype='int32')
            pred_classes = model.predict_classes(X)
            pred_proba = model.predict_proba(X)

            #print("Contagem rotulos: {0}".format(np.bincount(pred_classes)))
            rotulo_pred = np.argmax(np.bincount(pred_classes[np.where(pred_classes != NUM_CLASSES)]))
           
            prob_pred = np.max(pred_proba[np.where(pred_classes == rotulo_pred)])
            #print("Imagem {0} \n Predicao: {1} - Real: {2}".format(arq_img, rotulo_pred, rotulo))
        
            imagens.append({'id': id_img, 'arquivo': arq_img, 'real': rotulo, 'pred': rotulo_pred, 'prob_pred':prob_pred, 'patches': patches})
        
        labels = ['benigno','maligno']    
        y_pred = []
        y_true = []
        y_scores = []
        for r in imagens:
            y_pred.append(r['pred'])
            y_true.append(r['real'])
            y_scores.append(r['prob_pred'])
    
        y_pred_l = [labels[i] for i in y_pred]
        y_true_l = [labels[i] for i in y_true]
        cf_matrix = confusion_matrix(y_true_l, y_pred_l, labels=labels)
        tn, fp, fn, tp = cf_matrix.ravel()
        print("Matriz de Confusao:")
        print(str(cf_matrix))
        print(classification_report(y_true_l, y_pred_l, target_names=labels))
        print("Accuracy: {0}".format((tp+tn)/(tn+tp+fp+fn)))
        print("AUC (ROC): {}".format(roc_auc_score(y_true, y_scores)))

        fp_rate, tp_rate, thresholds = roc_curve(y_true, y_pred)
        print("AUC: {}".format(auc(fp_rate, tp_rate)))

        #fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fp_rate, tp_rate))
        tprs[-1][0] = 0.0
        roc_auc = auc(fp_rate, tp_rate)
        aucs.append(roc_auc)
        plt.plot(fp_rate, tp_rate, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (num_fold, roc_auc))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'ROC Media (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROCs - 5 folds - 400X')
    plt.legend(loc="lower right")
    plt.savefig('plots/todos-folds-fase01-400X.png')
    #plt.show()

if __name__ == '__main__':
    main()
