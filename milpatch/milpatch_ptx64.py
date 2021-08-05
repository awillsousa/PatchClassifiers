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

from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras import optimizers
from datetime import time, datetime
from random import shuffle

import os
import sys
import fnmatch
import argparse
import numpy as np
import mahotas as mh
import util
import ConfVars

FLAGS = None
NUM_CLASSES = 2
# Dimensao das imagens
SUBIMG_SIZE = 90
IMG_WIDTH, IMG_HEIGHT = 460, 700
PTX_DELTA=32
PTX_WIDTH = PTX_HEIGHT = PTX_SIZE = 64
EPOCHS = 100
BATCH_SIZE = 25
_EPSILON = K.epsilon()
OPT_ADAM = optimizers.Adam(lr=0.01)
OPT_RMSPROP = optimizers.rmsprop(lr=0.001, decay=1e-6)
#OPT_SGD = optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
OPT_SGD = optimizers.SGD(lr=0.00001, decay=4e-5, momentum=0.9, nesterov=True)
LOSS='mse'
OPTIMIZER=OPT_SGD

if K.image_data_format() == 'channels_first':
    INPUT_SHAPE = (3, PTX_WIDTH, PTX_HEIGHT)
else:
    INPUT_SHAPE = (PTX_WIDTH, PTX_HEIGHT, 3)


def main():
    print(str(FLAGS))
    # Gera uma lista das imagens de treino
    print("Carregando imagens e extraindo patches ...")
    
    # Dados de treino
    arqs_tr = util.lista_imgs(dir_imagens=FLAGS.images_tr, arquivo_labels=FLAGS.labels_tr)    
    x_treino, y_treino, imagens_tr = util.prep_bases2(arqs_tr, PTX_SIZE, PTX_DELTA, gera_dic=True, base=FLAGS.base)
    x_treino = np.asarray(x_treino)
    y_treino = np.asarray(y_treino)
    y_treino = to_categorical(y_treino, NUM_CLASSES)
    print ("Amostras de treino: %d"%x_treino.shape[0])
    
    # Dados de teste
    arqs_ts = util.lista_imgs(FLAGS.images_ts, FLAGS.labels_ts)        
    x_teste, y_teste, imagens_ts = util.prep_bases2(arqs_ts, PTX_SIZE, PTX_DELTA, gera_dic=False, base=FLAGS.base)
    x_teste = np.asarray(x_teste)
    y_teste = np.asarray(y_teste)     
    y_teste = to_categorical(y_teste, NUM_CLASSES)
    print ("Amostras de teste: %d"%x_teste.shape[0])
    
    EXISTE_VAL = False
    # Dados de validacao    
    if FLAGS.images_val != '':
        arqs_val = util.lista_imgs(FLAGS.images_val, FLAGS.labels_val)    
        x_val, y_val, imagens_val = util.prep_bases2(arqs_val, PTX_SIZE, PTX_DELTA, gera_dic=False, base=FLAGS.base)
        x_val = np.asarray(x_val)
        y_val = np.asarray(y_val)     
        y_val = to_categorical(y_val, NUM_CLASSES)
        EXISTE_VAL = True
        print ("Amostras de validacao: %d"%x_val.shape[0])    
        
    print ("Numero de classes: %d"%NUM_CLASSES)
           
    # Recupera as informacoes de data e hora para usar
    # nos arquivos de log 
    dirtime = datetime.now().strftime('%Y%m%d-%H%M%S')
     
    print("Iniciando fase 1 ...")    
    modelos_acuracia = []    
    #Cria o modelo a ser treinado  
    arquivo_pesos = os.path.join(FLAGS.model)
    if os.path.isfile(arquivo_pesos):    
        print("Carregando modelo EXISTENTE...")
        model = util.model_fase01(INPUT_SHAPE, NUM_CLASSES, arquivo_pesos=arquivo_pesos)
        model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])        
    else:    
        print("Criando um NOVO modelo...")
        model = util.model_fase01(INPUT_SHAPE, NUM_CLASSES)        
        model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])        
        
        # checkpoint
        outputFolder = './checkpoints'
        # apaga arquivos de modelos existentes
        for raiz, diretorios, arquivos in os.walk(outputFolder):
            for arquivo in arquivos:
                if arquivo.endswith('fase01.h5'):
                    os.remove(os.path.join(raiz,arquivo))

        arq_modelo = outputFolder+"/pesos-{epoch:02d}-{val_acc:.2f}-fase01.h5"
        checkpoint = ModelCheckpoint(arq_modelo, monitor='val_acc', 
                                     verbose=1, save_best_only=False, 
                                     save_weights_only=True, mode='auto', 
                                     period=5)    
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5, verbose=1, mode='auto')
        callbacks_list = [checkpoint, earlystop]

        # Treina o modelo
        start = datetime.now()        
        if EXISTE_VAL:
            model_info = model.fit(x_treino, y_treino, batch_size=64, 
                                   epochs=150, verbose=2, callbacks=callbacks_list, 
                                   validation_data=(x_val, y_val),
                                   shuffle=True)
        else:    
            model_info = model.fit(x_treino, y_treino, batch_size=64, 
                                   epochs=150, verbose=2, callbacks=callbacks_list, 
                                   validation_split=0.2)
        delta = datetime.now() - start        
        
        # Comentado: utilizar apenas os checkpoints salvos
        model.save_weights(outputFolder + '/pesos-train-ultimo-fase01.h5')
        
        # Exibe informações do treinamento
        util.plot_model_history(model_info, file='fase01-train-'+str(dirtime)+'-mse-b64-e150')
        print ("Tempo de %0.2f s para treinar o modelo"%(delta.total_seconds()))

        # Avalia os resultados para todos os modelos treinados
        for raiz, diretorios, arquivos in os.walk(outputFolder):
            for arquivo in arquivos:
                if arquivo.endswith('fase01.h5'):                    
                    model = util.model_fase01(INPUT_SHAPE, NUM_CLASSES, arquivo_pesos=os.path.join(raiz,arquivo))        
                    acuracia = util.accuracy(x_teste, y_teste, model)
                    modelos_acuracia.append((os.path.join(raiz,arquivo), acuracia))
                    print ("Modelo {0} - Acurácia nos dados de teste: {1:0.2f}".format(arquivo, acuracia))
        
        # Ordena os modelos pelos resultados de acuracia
        modelos_acuracia = sorted(modelos_acuracia, key=lambda k: k[1])
        
        
    print("Iniciando fase 2 ...")
    # Com o modelo treinado, selecionamos as duas classes de patches
    # Am => para cada imagem, o patch mais discriminativo
    # Bm => patches para os quais a CNN calculou altas probabilidades, 
    # mas com a classe errada e patches onde a entropia esteja acima de     
    # um determinado limiar
    modelos_fase02 = []
    i = -1
    testados = []   # lista de modelos da Fase2 que ja foram testados
    for arq_modelo, acuracia in modelos_acuracia[-5:]: 
        i += 1
        model = util.model_fase01(INPUT_SHAPE, NUM_CLASSES, arquivo_pesos=arq_modelo)
        model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])        
        
        Am = []
        Bm = []
        t_classes = [0,0,0] 
        total_imgs = len(imagens_tr)
        
        for i,img in enumerate(imagens_tr):
            print("Processando imagem {0}/{1}".format(i,total_imgs))
            for subimg in img['patches']:
                pred_probas = model.predict_proba(np.asarray([subimg['patch']]))
                classe = np.argmax(pred_probas)
                
                if pred_probas.shape[1] > 1:
                    delta = abs(pred_probas[0][0] - pred_probas[0][1])
                else:
                    delta = pred_probas[0][0]
            
                
                #print("Predicao: {0} Real: {1}".format(classe, patch['rotulo']))
                if subimg['rotulo'] != classe or delta < 0.3:                
                    subimg['rotulo'] = NUM_CLASSES # atribui o rotulo K+1 para esses patches
                    Bm.append(subimg)
                    t_classes[NUM_CLASSES] += 1
            
                else:
                    Am.append(subimg)
                    t_classes[classe] += 1
                    
        print("Total de exemplares por classe: {0}".format(str(t_classes)))            
        if False:#any(i == 0 for i in t_classes):
           print("Esse modelo está uma merda - treine um novo!")
           #sys.exit(0)
        else:
            modelos_fase02.append((i, arq_modelo, acuracia, t_classes)) # modelo, acuracia_fase01, total por classe, acuracia_fase02
            resultados = []            
            
            # Gera um novo conjunto de treinamento e testes
            X = []
            y = []
            L = Am+Bm
            shuffle(L)
            for p in L:
                X.append(p['patch'])
                y.append(p['rotulo'])
                    
            # Separa as bases de treino e de validacao
            x_treino = np.asarray(X)    
            y_treino = np.asarray(y)        
            
            y_treino = to_categorical(y_treino, NUM_CLASSES+1)
            
            model.layers.pop()
            model.layers.pop()            
            nova_fc = Dense(NUM_CLASSES+1, activation='softmax', name='novapredicao')    
            inp = model.input
            out = nova_fc(model.layers[-1].output)    
            model2 = Model(inp, out)
            
            model2.compile(loss=LOSS,
                          optimizer=OPTIMIZER,
                          metrics=['accuracy'])
            
            BATCH_SIZE = 64     
            tbCallback = TensorBoard(log_dir='./logs/'+str(i)+'-fase02-{0}'.format(dirtime), histogram_freq=0, 
                            batch_size=BATCH_SIZE, write_graph=True, 
                            write_grads=False, write_images=False, 
                            embeddings_freq=0, embeddings_layer_names=None, 
                            embeddings_metadata=None)
        
            early_stop = EarlyStopping(monitor='val_loss',
                                      min_delta=0.001,
                                      patience=5,
                                      verbose=0, mode='auto')         
            outputFolder = './checkpoints'
            filepath = outputFolder + "/"+str(i)+"-pesos-{epoch:02d}-{val_acc:.2f}-fase02.h5"
            checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                                         save_best_only=False, save_weights_only=True, 
                                         mode='auto', period=5)
            callbacks_list = [tbCallback, early_stop, checkpoint]        
            
            # Treina o modelo
            start = datetime.now()
            model_info = model2.fit(x_treino, y_treino, batch_size=64, 
                                   epochs=150, verbose=2, callbacks=callbacks_list, 
                                   validation_split=0.2)
            delta = datetime.now() - start        
            
            # Exibe informações do treinamento
            print("Modelo da Fase1: {}".format(arq_modelo))
            nome_plot = arq_modelo.split('/')[-1]
            nome_plot = nome_plot.split('.')[-2]
            #util.plot_model_history(model_info, file=str(i)+'-fase02-train-'+str(dirtime)+'-mse-b64-e150')
            util.plot_model_history(model_info, file=str(i)+'-fase02-'+nome_plot)
            print ("Tempo de %0.2f s para treinar o modelo"%(delta.total_seconds()))
            
            # Comentado: utilizar apenas os checkpoints salvos
            #model2.save_weights(outputFolder + "/"+str(i)+'-pesos-train-ultimo-fase02.h5')

            # Avalia os modelos criados na Fase02
            # Avalia os resultados para todos os modelos treinados
            for raiz, diretorios, arquivos in os.walk(outputFolder):
                for arquivo in arquivos:
                    if arquivo.endswith('fase02.h5') and not(arquivo in testados):
                        model2 = util.model_fase01(INPUT_SHAPE, NUM_CLASSES+1, arquivo_pesos=os.path.join(raiz,arquivo))        
                        acuracia = util.accuracy(x_teste, y_teste, model2)
                        resultados.append((os.path.join(raiz,arquivo), acuracia))                        
                        modelos_fase02[-1] = (*modelos_fase02[-1], resultados)
                        testados.append(arquivo)
                        print ("Modelo {0} - Acurácia nos dados de teste: {1:0.2f}".format(arquivo, acuracia))                       
            
            # Ordena os modelos pelos resultados de acuracia
            #modelos_acuracia = sorted(modelos_acuracia, key=lambda k: k[1])
    
    

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--images_tr',
      type=str,
      default='',
      help='Diretorio das imagens de treino'
  )
  parser.add_argument(
      '--labels_tr',
      type=str,
      default=None,
      help='Arquivo contendo os rotulos das imagens de treino'
  )
  parser.add_argument(
      '--images_ts',
      type=str,
      default='',
      help='Diretorio das imagens de teste'
  )
  parser.add_argument(
      '--labels_ts',
      type=str,
      default=None,
      help='Arquivo contendo os rotulos das imagens de teste'
  )
  parser.add_argument(
      '--images_val',
      type=str,
      default='',
      help='Diretorio das imagens de validacao'
  )
  parser.add_argument(
      '--labels_val',
      type=str,
      default=None,
      help='Arquivo contendo os rotulos das imagens de validacao'
  )
  parser.add_argument(
      '--base',
      type=str,
      default='cancer',
      help='Base de imagens utilizada'
  )
  parser.add_argument(
      '--output_graph',
      type=str,
      default='./models/output_graph.pb',
      help='Diretorio de saida do modelo gerado.'
  ) 
  parser.add_argument(
      '--training_steps',
      type=int,
      default=150,
      help='Total de execucoes do treinamento.'
  )
  parser.add_argument(
      '--model',
      type=str,
      default='pesos-fase01.h5',
      help="Nome do arquivo de modelo em ./models/"
  )
  '''
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='Learning rate.'
  )
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='Percentual de imagens para utilizar como teste.'
  )
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='Percentual de imagens para utilizar como validacao.'
  )  
  parser.add_argument(
      '--train_batch_size',
      type=int,
      default=100,
      help='Quantas imagens treinar de cada vez.'
  )    
  
  '''
  FLAGS, unparsed = parser.parse_known_args()
  main()
