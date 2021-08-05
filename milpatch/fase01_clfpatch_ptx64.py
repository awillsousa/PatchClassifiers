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
import util
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os



FLAGS = None
NUM_CLASSES = 2

# Dimensao das imagens
IMG_WIDTH, IMG_HEIGHT = 460, 700
PTX_WIDTH = PTX_HEIGHT = PTX_SIZE = 64
EPOCHS = 150
BATCH_SIZE = 64
OPTIMIZER='adam'
PTX_DELTA=32

if K.image_data_format() == 'channels_first':
    INPUT_SHAPE = (3, PTX_WIDTH, PTX_HEIGHT)
else:
    INPUT_SHAPE = (PTX_WIDTH, PTX_HEIGHT, 3)


def main():
    
    #Cria o modelo a ser treinado  
    arquivo_pesos = os.path.join(FLAGS.model)
    
    if os.path.isfile(arquivo_pesos):    
        model = util.model_fase01(INPUT_SHAPE, NUM_CLASSES, arquivo_pesos)
    else:    
        print("Modelo não encontrado!")
        os.sys.exit(1)
    
    # Gera uma lista das imagens a classificar   
    arqs_imgs = util.lista_imgs(FLAGS.image_dir, FLAGS.labels)
    
    #arqs_imgs = [os.path.join(FLAGS.image_dir,f) for f in os.listdir(FLAGS.image_dir) if fnmatch.fnmatch(f,'*.png')]
    #arqs_imgs += [os.path.join(FLAGS.image_dir,f) for f in os.listdir(FLAGS.image_dir) if fnmatch.fnmatch(f,'*.jpg')]
    
    # Gera patches a partir das imagens originais. 
    # A classificacao das imagens sera feita em nivel de patch
    imagens = []
    # Itera todas as imagens do conjunto de treino
    for id_img,arq_img in enumerate(arqs_imgs):
        X_1 = []
        y_1 = []
        print("Extraindo patches da imagem {0}".format(arq_img))        
        imagem = util.carrega_imagem(arq_img)        
        rotulo = util.get_label(arq_img, base=FLAGS.base)
        
        # Extrai subimagens de tamanho 90x90 com deslocamento de 60
        # Como as imagens são 800x600 e para imagem somente sera extraido o patch mais discriminante,
        # subdividi as imagens para termos mais patches selecionados por imagem
        patches = util.gera_patches(imagem, arq_img, rotulo, tamanho=PTX_SIZE, desloca=PTX_DELTA)
        
        for p in patches:
            X_1.append(p['patch'])
            y_1.append(p['rotulo'])
        
        X = np.asarray(X_1, dtype='float32')
        y = np.asarray(y_1, dtype='int32')
        pred_classes = model.predict_classes(X)
        pred_proba = model.predict_proba(X)
        
        print("Shape probas: "+str(pred_proba.shape))
        print("Shape classes: "+str(pred_classes.shape))
        
        # Predicao utilizando maximos        
        rotulo_pred = np.argmax(np.bincount(pred_classes))
        prob_pred = np.max(pred_proba[np.where(pred_classes == rotulo_pred)])
        #prob_pred = np.max(np.max(pred_proba, axis=0))
        #print(str(np.argmax(np.max(pred_proba, axis=0))))        
        #print(str(np.max(pred_proba, axis=0)))
        
        
        # Predicao utilizando medias
        #medios_proba = np.mean(pred_proba[np.where(pred_classes != NUM_CLASSES)], axis=0)
        #rotulo_pred = np.argmax(medios_proba)        
        #prob_pred = np.max(medios_proba)
        
        print("Imagem: {0} Rotulo: {1} Predicao: {2} Probabilidade: {3}".format(arq_img, rotulo, rotulo_pred, prob_pred))
        imagens.append({'id': id_img, 'arquivo': arq_img, 'real': rotulo, 'pred': rotulo_pred, 'prob_pred':prob_pred, 'patches': patches})
        
    
    #labels = ['benigno','maligno']    
    labels = ConfVars.LABELS[FLAGS.base]    
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
    acuracia = (tp+tn)/(tn+tp+fp+fn)
    print("Accuracy: {0}".format(acuracia))
    print(classification_report(y_true_l, y_pred_l, target_names=labels))
    print("AUC (ROC): {}".format(roc_auc_score(y_true, y_scores)))
    fp_rate, tp_rate, thresholds = roc_curve(y_true, y_pred)
    valor_auc = auc(fp_rate, tp_rate)
    print("AUC: {}".format(valor_auc))
        
    # Plota as curvas de erro x rejeicao
    taxas_erro = [1-acuracia]
    taxas_rejeicao = [0]
    taxas_rec = [acuracia]
    taxas_rel = [acuracia]
    
    l_prob = min(y_scores)
    h_prob = max(y_scores)
    
    print("Probabilidade Min: {}".format(l_prob))
    print("Probabilidade Max: {}".format(h_prob))
        
    limiar = l_prob
    inc = 0.01
    erro = 1-acuracia
    total_imagens = len(imagens)
    while erro > 0 and limiar <= h_prob:
        limiar += inc        
            
        y_pred = []
        y_true = []
        y_scores = []
        rejeitados = 0
        recs = 0
        errados = 0
        
        for r in imagens:
          if r['prob_pred'] > limiar:
              if r['pred'] == r['real']:
                  recs += 1
              else:
                  errados += 1              
          else:
              rejeitados += 1
        
        rec = recs/total_imagens        
        erro = errados/total_imagens
        rejeicao = rejeitados/total_imagens
        
        if rec == 0 and erro == 0:
            rel = 0
        else:
            rel = rec/(rec+erro)
        
        taxas_rec.append(rec)
        taxas_erro.append(erro)
        taxas_rejeicao.append(rejeicao)
        taxas_rel.append(rel)
        
        print("Reconhecimento: {0} Erro: {1} Rejeicao: {2} Confiabilidade: {3}".format(rec, erro, rejeicao, rel))
        
        
    # Plota curva erro x rejeicao    
    '''
    plt.figure()
    lw = 1
    plt.plot(taxas_rejeicao, taxas_erro, color='navy',
            label=r'Para erro = %0.2f AUC = %0.2f ACC = %0.2f)' % (taxas_erro[0], valor_auc, taxas_rec[0]),
                                          lw=2, alpha=.8)
        
    #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, max(taxas_erro)+.05])
    plt.xlabel('% Rejeicao')
    plt.ylabel('% Erro')
    plt.title('Erro x Rejeicao')
    plt.legend(loc="upper center", shadow=True, fontsize='large')
    #plt.show()
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    linha1 = ax.plot(taxas_rejeicao, taxas_erro, '-', label = 'Erro')
    ax.annotate('AUC: {1:0.2f} ACC: {0:0.2f}'.format(taxas_rec[0], valor_auc), xy=(0,taxas_erro[0]), xytext=(0.2, taxas_erro[0]+0.1), arrowprops=dict(facecolor='black', shrink=0.05))
    ax2 = ax.twinx()
    linha2 = ax2.plot(taxas_rejeicao, taxas_rel, '-r', label = 'Confiabilidade')
    ax.legend(shadow=True, fontsize='large', loc=2)
    ax2.legend(shadow=True, fontsize='large', loc=1)
    ax.grid()
    ax.set_xlabel("% Rejeição")
    ax.set_ylabel(r"% Erro")
    ax2.set_ylabel(r"% Confiabilidade")
    ax2.set_ylim(0,1.0)
    ax.set_ylim(0,1.0)
    ax.set_xlim(0,1.0)
    plt.savefig("./plots/erro_x_rejeicao_"+FLAGS.base+"_"+FLAGS.fold+".png")
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--image_dir',
      type=str,
      default='',
      help='Diretorio das imagens para classificar'
  )  
  parser.add_argument(
      '--labels',
      type=str,
      default=None,
      help='Arquivo contendo rotulos das imagens'
  )  
  parser.add_argument(
      '--model',
      type=str,
      default='./models/pesos-fase02.h5',
      help="Pesos do modelo treinado"
  )
  parser.add_argument(
      '--base',
      type=str,
      default='cancer',
      help="Base de imagens a classificar. Valores: cancer, especies, dtd"
  )
  
  parser.add_argument(
      '--fold',
      type=str,
      default='',
      help="Indicador de qual fold esta sendo processado"
  )
  FLAGS, unparsed = parser.parse_known_args()
  main()
