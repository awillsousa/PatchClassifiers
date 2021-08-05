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

from keras.utils import to_categorical, np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras import optimizers
from datetime import time, datetime
from random import shuffle

import sys
import util
import fnmatch
import argparse
import ConfVars
import numpy as np
import mahotas as mh
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os

# DEFINICAO DE CONSTANTES E CONFIGURAÇÕES
FLAGS = None
NUM_CLASSES = 2
# Dimensao das imagens
SUBIMG_SIZE = 90
SUBIMG_DELTA = 60
IMG_WIDTH, IMG_HEIGHT = 460, 700
PTX_WIDTH = PTX_HEIGHT = PTX_SIZE = 50
PTX_DELTA = 25
EPOCHS = 100
BATCH_SIZE = 25
_EPSILON = K.epsilon()

MODELO='./models/pesos-fase01.h5'

if K.image_data_format() == 'channels_first':
    INPUT_SHAPE = (3, PTX_WIDTH, PTX_HEIGHT)
else:
    INPUT_SHAPE = (PTX_WIDTH, PTX_HEIGHT, 3)


##############################################################################################################

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


# Testa o modelo
def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class) 
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)


def get_loss():
    def loss(y_true, y_pred):                          
            y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON) 
            m = -K.log(K.max(y_true*y_pred, axis=-1))
            #K.tf.add_to_collection('losses', -K.log(m+_EPSILON))
            #l = K.tf.add_n(K.tf.get_collection('losses'))
            l = K.sum(m)
            return (l)
    return loss

def get_loss2():
    def loss(y_true, y_pred):                           
            m = K.max(K.tf.multiply(y_true+_EPSILON, y_pred+_EPSILON), axis=1)            
            l = -K.sum(K.log(m), axis=-1)            
            return (l)
    return loss

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


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
            patches.append({'id':i, 'arquivo': arquivo, 'tam': (tamanho,tamanho), 'pos': posicao, 'patch': p, 'rotulo': rotulo, 'subpatches': []})  
            i+=1
            
    #print("Extraidos {0} patches".format(str(len(patches))))
    
    return (patches)

##############################################################################################################

def main():

    dirtime = datetime.now().strftime('%Y%m%d-%H%M%S')
    np.random.seed(2017)
    
    '''
    Faz a carga e divisao das bases 
    '''
    
    arq_imgs = util.lista_imgs(FLAGS.image_tr)
    x_treino, y_treino, imagens = util.prep_bases2(arq_imgs, PTX_SIZE, PTX_DELTA, gera_dic=False)
    
    arq_ts = util.lista_imgs(FLAGS.image_ts)
    x_teste, y_teste, imagens = util.prep_bases2(arq_ts, PTX_SIZE, PTX_DELTA, gera_dic=False)
    
    nc = int(len(x_treino)*0.8)
    #x_treino = np.asarray(x_treino, dtype='float64')
    x_treino = np.asarray(x_treino)
    #x_treino = np.true_divide(x_treino, 255)
    y_treino = np.asarray(y_treino)
    #x_teste = np.asarray(x_teste, dtype='float64')
    x_teste = np.asarray(x_teste)
    #x_teste = np.true_divide(x_teste, 255)
    y_teste = np.asarray(y_teste)

    print('Formato x_treino:', x_treino.shape)
    print(x_treino.shape[0], 'amostras de treino')
    print(x_teste.shape[0], 'amostras de teste')
    
    print ("Amostras de treino: %d"%x_treino.shape[0])
    print ("Amostras de teste: %d"%x_teste.shape[0])
    print ("Number of classes: %d"%NUM_CLASSES)
    
    # Imprime alguns exemplares da base
    fig = plt.figure(figsize=(8,3))
    for i in range(NUM_CLASSES):
        ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
        features_idx = x_treino[y_treino[:]==i,:]
        ax.set_title("Num: " + str(i))
        plt.imshow(features_idx[1], cmap="gray")
    plt.savefig("plots/exemplos.png")    
    #plt.show()
    
    # Preprocessamento
    
    # Altera o formato das matrizes 
    #x_treino = x_treino.reshape(x_treino.shape[0], img_rows*img_cols)
    #x_teste = x_teste.reshape(x_teste.shape[0], img_rows*img_cols)
    
    # Converte rotulos pra one-hot encoding
    y_treino = to_categorical(y_treino, NUM_CLASSES)
    y_teste = to_categorical(y_teste, NUM_CLASSES)
   
    print("########### 1º MODELO DE TESTE ###########")
 
    # Define o modelo
    model = util.model_fase01(INPUT_SHAPE, NUM_CLASSES, arquivo_pesos=MODELO)
    
    # Define o otimizador
    sgd = optimizers.SGD(lr=0.01)
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
    
    # Exibe informações do modelo
    model.summary()
    
    # Treinar o modelo
    start = datetime.now()
    model_info = model.fit(x_treino, y_treino, batch_size=64, nb_epoch=10, verbose=2, validation_split=0.2)
    delta = datetime.now() - start
    print ("Tempo de %0.2f s para treinar o modelo"%(delta.total_seconds()))
    model.save_weights('./models/sgd-lr001-mse-b64-e10.h5')
    
    # Plota os dados do historico do treinamento    
    plot_model_history(model_info, file='sgd-lr001-mse-b64-e10')
    
    print ("Acurácia nos dados de teste: %0.2f"%accuracy(x_teste, y_teste, model))
    
    # Variando o learning rate
    # Decrementa o learning rate
   
    print("########### 2º MODELO DE TESTE ###########")
    
    # Define um novo modelo
    model = util.model_fase01(INPUT_SHAPE, NUM_CLASSES, arquivo_pesos=MODELO)
    
    # Define o optimizador
    sgd = optimizers.SGD(lr=0.001)
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
    
    # Treina o modelo
    start = datetime.now()
    model_info = model.fit(x_treino, y_treino, batch_size=64, nb_epoch=10, verbose=2, validation_split=0.2)
    delta = datetime.now() - start
    model.save_weights('./models/sgd-lr0001-mse-b64-e10.h5')
    
    # Exibe informações do treinamento
    plot_model_history(model_info, file='sgd-lr0001-mse-b64-e10')
    print ("Tempo de %0.2f s para treinar o modelo"%(delta.total_seconds()))

    
    # Avalia o modelo
    print ("Acurácia nos dados de teste: %0.2f"%accuracy(x_teste, y_teste, model))
    
    #Incrementando o learning rate
    
    print("########### 3º MODELO DE TESTE ###########")
    
    # Incrementa o learning rate
    # Define um novo modelo
    model = util.model_fase01(INPUT_SHAPE, NUM_CLASSES, arquivo_pesos=MODELO)
    
    # Define o optimizador
    sgd = optimizers.SGD(lr=0.1)
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
    
    # Treina o modelo
    start = datetime.now()
    model_info = model.fit(x_treino, y_treino, batch_size=64, nb_epoch=10, verbose=2, validation_split=0.2)
    delta = datetime.now() - start
    model.save_weights('./models/sgd-lr01-mse-b64-e10.h5')
    # Exibe informações do treinamento
    plot_model_history(model_info, file='sgd-lr01-mse-b64-e10')
    print ("Tempo de %0.2f s para treinar o modelo"%(delta.total_seconds()))
    
    # Avalia o modelo
    print ("Acurácia nos dados de teste: %0.2f"%accuracy(x_teste, y_teste, model))
    
    # Usando Adam ao inves de SGD
    
    print("########### 4º MODELO DE TESTE ###########")
    
    # Define um novo modelo
    model = util.model_fase01(INPUT_SHAPE, NUM_CLASSES, arquivo_pesos=MODELO)
    
    # Define o optimizador
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
    # Treina o modelo
    start = datetime.now()
    model_info = model.fit(x_treino, y_treino, batch_size=64, nb_epoch=10, verbose=2, validation_split=0.2)
    delta = datetime.now() - start
    model.save_weights('./models/adam-mse-b64-e10.h5')
    # Exibe informações do treinamento
    plot_model_history(model_info, file='adam-mse-b64-e10')
    print ("Tempo de %0.2f s para treinar o modelo"%(delta.total_seconds()))
    
    # Avalia o modelo
    print ("Acurácia nos dados de teste: %0.2f"%accuracy(x_teste, y_teste, model))
    
    # Variando o tamanho do batch
    # Incrementa o tamanho do batch
    
    print("########### 5º MODELO DE TESTE ###########")

    # Define um novo modelo
    model = util.model_fase01(INPUT_SHAPE, NUM_CLASSES, arquivo_pesos=MODELO)
    
    # Define o optimizador
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
    # Treina o modelo
    start = datetime.now()
    model_info = model.fit(x_treino, y_treino, batch_size=128, nb_epoch=10, verbose=0, validation_split=0.2)
    delta = datetime.now() - start
    model.save_weights('./models/adam-mse-b128-e10.h5')
    # Exibe informações do treinamento
    plot_model_history(model_info, file='adam-mse-b128-e10')
    print ("Tempo de %0.2f s para treinar o modelo"%(delta.total_seconds()))
    
    # Avalia o modelo
    print ("Acurácia nos dados de teste: %0.2f"%accuracy(x_teste, y_teste, model))
    
    # Decrementando o tamanho do baatch
    
    print("########### 6º MODELO DE TESTE ###########")
    
    # Define um novo modelo
    model = util.model_fase01(INPUT_SHAPE, NUM_CLASSES, arquivo_pesos=MODELO)
    
    # Define o optimizador
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
    # Treina o modelo
    start = datetime.now()
    model_info = model.fit(x_treino, y_treino, batch_size=32, nb_epoch=10, verbose=0, validation_split=0.2)
    delta = datetime.now() - start
    model.save_weights('./models/adam-mse-b32-e10.h5')
    # Exibe informações do treinamento
    plot_model_history(model_info, file='adam-mse-b32-e10')
    print ("Tempo de %0.2f s para treinar o modelo"%(delta.total_seconds()))
    
    # Avalia o modelo
    print ("Acurácia nos dados de teste: %0.2f"%accuracy(x_teste, y_teste, model))
 
    print("########### 7º MODELO DE TESTE ###########")
    
    # Mudando a funcao de custo
    # Define um novo modelo
    model = util.model_fase01(INPUT_SHAPE, NUM_CLASSES, arquivo_pesos=MODELO)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Treina o modelo
    start = datetime.now()
    model_info = model.fit(x_treino, y_treino, batch_size=128, nb_epoch=10, verbose=0, validation_split=0.2)
    delta = datetime.now() - start
    model.save_weights('./models/adam-catxentropy-b128-e10.h5')
    # Exibe informações do treinamento
    plot_model_history(model_info, file='adam-catxentropy-b128-e10')
    print ("Tempo de %0.2f s para treinar o modelo"%(delta.total_seconds()))
    
    # Avalia o modelo
    print ("Acurácia nos dados de teste: %0.2f"%accuracy(x_teste, y_teste, model))
    
    
    print("########### 8º MODELO DE TESTE ###########")
    
    # Aumentando o numero de epocas
     
    # define model
    model = util.model_fase01(INPUT_SHAPE, NUM_CLASSES, arquivo_pesos=MODELO)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Treina o modelo
    start = datetime.now()
    model_info = model.fit(x_treino, y_treino, batch_size=128, nb_epoch=100, verbose=0, validation_split=0.2)
    delta = datetime.now() - start
    model.save_weights('./models/adam-catxentropy-b128-e100.h5')
    # Exibe informações do treinamento
    plot_model_history(model_info, file='adam-catxentropy-b128-e100')
    print ("Tempo de %0.2f s para treinar o modelo"%(delta.total_seconds()))
    
    # Avalia o modelo
    print ("Acurácia nos dados de teste: %0.2f"%accuracy(x_teste, y_teste, model))
    
    # checkpoint
    outputFolder = './checkpoints'
    #if not os.path.isdir(outputFolder):
    #    os.makedirs(outputFolder)
    filepath=outputFolder+"/weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=10)

    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=1, mode='auto')
    callbacks_list = [checkpoint, earlystop]
    
    # train the model
    model_info = model.fit(x_treino, y_treino, batch_size=128, \
                           nb_epoch=80, callbacks=callbacks_list, verbose=0, \
                           validation_split=0.2)
    # compute test accuracy
    print ("Acurácia nos dados de teste: %0.2f"%accuracy(x_teste, y_teste, model))
    ''' 
    # Resume o treinamento e carrega um checkpoint
    
    # define model
    model = util.model_fase01(INPUT_SHAPE, NUM_CLASSES)
    
    
    # load weights
    import os, glob
    epoch_num = 79
    outputFolder = './checkpoints'
    file_ini = outputFolder+'/weights-'+ str(epoch_num)+'*'
    filename =  glob.glob(file_ini)
    if os.path.isfile(filename[0]):
        model.load_weights(filename[0])
    else:
        print ("%s não existe"%filename[0])
        
    # Define o optimizador
    sgd = optimizers.SGD(lr=0.1)
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
    
    # Checkpoint
    outputFolder = './checkpoints'
    #if not os.path.isdir(outputFolder):
    #    os.makedirs(outputFolder)
        
    filepath=outputFolder+"/weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=10)
    callbacks_list = [checkpoint]
    
    # Definicao de EarlyStop
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=1, mode='auto')
    callbacks_list = [earlystop]
    
    # train the model
    start = datetime.now()
    model_info = model.fit(x_treino, y_treino, batch_size=128, nb_epoch=100, callbacks=callbacks_list, verbose=0, validation_split=0.2)
    end = datetime.now()
    
    # Exibe informações de treinamento
    plot_model_history(model_info, file='ckpt-sgd-lr01-b128-e100')
    
    # Resultados na base de teste
    print ("Acurácia nos dados de teste: %0.2f"%accuracy(x_teste, y_teste, model))
    '''


##############################################################################################################

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--image_tr',
      type=str,
      default='',
      help='Diretorio das imagens de treino'
  )

  parser.add_argument(
      '--image_ts',
      type=str,
      default='',
      help='Diretorio das imagens de treino'
  )
      
  FLAGS, unparsed = parser.parse_known_args()
  main()
    
