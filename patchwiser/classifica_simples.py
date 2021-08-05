# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 13:34:10 2016
@author: antoniosousa
Realiza o processo de classificacao de imagens baseado no processo
de extracao de patches das imagens.

"""

from os import path
from time import time
from optparse import OptionParser
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import sys
import numpy as np
import extrator as ex
import matplotlib.pyplot as plt
import extrai
import helper
import logging 
from datetime import datetime


# Constantes   
SVM_C = 32  # parametro C a ser utilizado para a classificacao usando SVM
SVM_G = 0.5 # paramatro gama a ser utilizada para a classificacao usando SVM

# lista de classificadores
CLFS = {'knn':('KNN', KNeighborsClassifier(3)), 
        'svm':('SVM', SVC(gamma=0.5, C=32, cache_size=250, probability=True)),
        'dt':('Árvore de Decisão', DecisionTreeClassifier(max_depth=5)),
        'rf':('Random Forest', RandomForestClassifier(max_features=0.1, n_estimators=500, min_samples_leaf=0.01, n_jobs=3)),
        'qda':('QDA', QuadraticDiscriminantAnalysis())}
FUSAO = ['voto', 'soma', 'produto', 'serie']     # metodos de fusao a serem utilizados
CLASSES = {'B':0, 'M':1}    # representacao numerica das classes 
DESCARTA = False


'''
Retorna um classificador (funcao) a ser utilizado
'''
def get_clf(nome_clf): 
    c = CLFS[nome_clf]    
    return (c[1]) 

'''
Retorna a descrição (texto) do classificador
'''
def get_desc_clf(nome_clf):
    c = CLFS[nome_clf]     
    return (c[0])


'''
Executa a classificacao de uma imagem
imagem - caminho do arquivo
clf - classificador treinado com atributos da base de treinamento
atrib_ts - vetores de atributos dos patches da imagem
rotulos_ts - vetor de rotulos dos patches da imagem
'''
def classifica_img(imagem, clf, atrib_ts):
    logging.info("Classificacao imagem " + imagem)
    
    # recupera o rotulo real da imagem
    classe, _ = ex.classe_arquivo(imagem)                
    rotulo_real = ex.CLASSES[classe]
        
    # predicao do classificador para o conjunto de patches        
    ls_preds = clf.predict(atrib_ts)     
    ls_preds = np.asarray(ls_preds, dtype='int32')      

    # utiliza apenas as prediçoes dos patches cuja probabilidade esteja
    # acima do limiar   
    if DESCARTA: 
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(atrib_ts)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(atrib_ts)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    
        print("prob_pos: tam=" + str(len(prob_pos)))
        ls_preds = ls_preds[np.where(prob_pos > 0.90)]
        print("ls_preds: tam=" + str(len(ls_preds)))    
                          
    conta = np.bincount(ls_preds)
    logging.info('Contagem classes patches: ' + str(conta))
    
    rotulo_pred = np.argmax(conta)
    logging.info('Classe predição: ' + str(rotulo_pred))
        
    return (rotulo_real, rotulo_pred)
   
   
'''
Verifica se uma opção passada existe na lista de argumentos do parser
'''  
def existe_opt (parser, dest):
   if any (opt.dest == dest and (opt._long_opts[0] in sys.argv[1:] or opt._short_opts[0] in sys.argv[1:]) for opt in parser._get_all_options()):
      return True
   return False 
  
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
              'arquivo': l[5]           # caminho do arquivo da imagem
            }
        lista.append(d) 
    return (lista)
    
  
'''
Executa o procedimento de classificacao
Trabalha com o esquema de divisões sucessivas, efetuando a classificação
para cada conjunto de patches gerados por cada nivel das divisões efetudas
'''    
def classificacao_arquivo(base_tr, base_ts, arq_ppi, id_clf):
    inicio = time()    
    imagens = {}
    clf = get_clf(id_clf)    
    logging.info("<<<<<<<< classificacao_arquivo >>>>>>>>")    
    try: 
        # Carrega a base de treinamento      
        atrib_tr = None 
        rotulos_tr = None             
        atrib_tr, rotulos_tr = load_svmlight_file(base_tr, dtype=np.float32,n_features=162) 
        logging.info("Carregada a base de treinamento: " + base_tr)        
    
        # Treina o classificador
        logging.info("Treinando classificador...")
        clf.fit(atrib_tr, rotulos_tr)                    
        
        # Carrega a base de testes e o arquivo de patches por imagem                     
        atrib_ts = None 
        rotulos_ts = None             
        atrib_ts, rotulos_ts = load_svmlight_file(base_ts, dtype=np.float32, n_features=162)#, n_features=atrib_tr.shape[1]) 
        logging.info("Carregado arquivo da base de testes: " + base_ts)        
        
        # carrega arquivo de patches por imagem da base de teste        
        logging.info("Arquivo de patches: "+ arq_ppi)
        imagens = dicionario_imgs(helper.load_csv(arq_ppi))
        logging.info("Carregado arquivo de quantidade de patches por imagem: " + arq_ppi)
        logging.info("Classificando para " + id_clf )
        
        r_tst = []
        r_pred = []
        idx1 = 0    # posicao inicial dos atributos da imagem
        idx2 = 0    # posicao final dos atributos da imagem
        num_ppi = imagens[0]['total']
        total_desc = 0      # total de patches descartados
        
        # Carrega os atributos de acordo com as informações do arquivos de patches por imagem (.ppi)
        tempos_imgs = []
        for imagem in imagens:
            t0_imagem = time()            
            idx2 = imagem['ppi']            
            if idx2 > 0:
                idx2 += idx1  # limite superior da fatia                
                atribs_img = atrib_ts[idx1:idx2]                 
                tst,pred = classifica_img(imagem['arquivo'], clf, atribs_img)
                r_tst.append(tst)
                r_pred.append(pred)
                
                total_desc += imagem['descartados']                
                idx1 = idx2
            tempos_imgs.append(round(time()-t0_imagem,3))    
            logging.info("Tempo classificação imagem: " + str(tempos_imgs[-1]))
            
        # Loga estatisticas de tempo por imagem
        logging.info("Tempo medio de classificacao por imagem: {0}".format(np.mean(tempos_imgs)))
        logging.info("Desvio padrao tempo classificacao por imagem: {0}".format(np.std(tempos_imgs)))
        # cria as matrizes de confusao
        cm = confusion_matrix(r_tst, r_pred)
        
        # exibe a taxa de classificacao
        r_pred = np.asarray(r_pred)
        r_tst = np.asarray(r_tst)
        taxa_clf = np.mean(r_pred.ravel() == r_tst.ravel()) * 100
        logging.info("Taxa de Classificação: %f " % (round(taxa_clf,3)))     
        
        tempo_exec = time()-inicio        
        # armazena os resultados
        resultado = { 'ppi':num_ppi,               # patches utilizados por imagem
                      'descartados':total_desc,    # total de patches descartados
                      'total':imagens[0]['total'], # total de patches gerados para a imagem
                      'taxa_clf':round(taxa_clf,3),  # taxa de classificacao 
                      'tempo':round(tempo_exec,3),   # tempo de execucao da classificacao
                      'matriz':cm           # matriz de confusao
                    }
        
        return (resultado)
    except Exception as e:
        logging.info(str(e))

'''
Executa o procedimento de classificacao
Trabalha com o esquema de divisões sucessivas, efetuando a classificação
para cada conjunto de patches gerados por cada nivel das divisões efetudas
'''    
def classificacao_base(atrib_tr, rotulos_tr, base_ts, arq_ppi, id_clf):
    inicio = time()    
    imagens = {}
    clf = get_clf(id_clf)    
    logging.info("<<<<<<<< classificacao_arquivo >>>>>>>>")    
    try: 
        # Carrega a base de treinamento      
        if atrib_tr == None:
            loga_sai("Falha na carga da base de treinamento" )        
    
        # Treina o classificador
        logging.info("Treinando classificador...")
        clf.fit(atrib_tr, rotulos_tr)                    
        
        # Carrega a base de testes e o arquivo de patches por imagem                     
        atrib_ts = None 
        rotulos_ts = None             
        atrib_ts, rotulos_ts = load_svmlight_file(base_ts, dtype=np.float32, n_features=162)#, n_features=atrib_tr.shape[1]) 
        logging.info("Carregado arquivo da base de testes: " + base_ts)        
        
        # carrega arquivo de patches por imagem da base de teste        
        logging.info("Arquivo de patches: "+ arq_ppi)
        imagens = dicionario_imgs(helper.load_csv(arq_ppi))
        logging.info("Carregado arquivo de quantidade de patches por imagem: " + arq_ppi)
        logging.info("Classificando para " + id_clf )
        
        r_tst = []
        r_pred = []
        idx1 = 0    # posicao inicial dos atributos da imagem
        idx2 = 0    # posicao final dos atributos da imagem
        num_ppi = imagens[0]['total']
        total_desc = 0      # total de patches descartados
        
        # Carrega os atributos de acordo com as informações do arquivos de patches por imagem (.ppi)
        tempos_imgs = []
        for imagem in imagens:
            t0_imagem = time()            
            idx2 = imagem['ppi']            
            if idx2 > 0:
                idx2 += idx1  # limite superior da fatia                
                atribs_img = atrib_ts[idx1:idx2]                 
                tst,pred = classifica_img(imagem['arquivo'], clf, atribs_img)
                r_tst.append(tst)
                r_pred.append(pred)
                
                total_desc += imagem['descartados']                
                idx1 = idx2
            tempos_imgs.append(round(time()-t0_imagem,3))    
            logging.info("Tempo classificação imagem: " + str(tempos_imgs[-1]))
            
        # Loga estatisticas de tempo por imagem
        logging.info("Tempo medio de classificacao por imagem: {0}".format(np.mean(tempos_imgs)))
        logging.info("Desvio padrao tempo classificacao por imagem: {0}".format(np.std(tempos_imgs)))
        # cria as matrizes de confusao
        cm = confusion_matrix(r_tst, r_pred)
        
        # exibe a taxa de classificacao
        r_pred = np.asarray(r_pred)
        r_tst = np.asarray(r_tst)
        taxa_clf = np.mean(r_pred.ravel() == r_tst.ravel()) * 100
        logging.info("Taxa de Classificação: %f " % (round(taxa_clf,3)))     
        
        tempo_exec = time()-inicio        
        # armazena os resultados
        resultado = { 'ppi':num_ppi,               # patches utilizados por imagem
                      'descartados':total_desc,    # total de patches descartados
                      'total':imagens[0]['total'], # total de patches gerados para a imagem
                      'taxa_clf':round(taxa_clf,3),  # taxa de classificacao 
                      'tempo':round(tempo_exec,3),   # tempo de execucao da classificacao
                      'matriz':cm           # matriz de confusao
                    }
        
        return (resultado)
    except Exception as e:
        logging.info(str(e))


# Executa o processo de classificação invocando a fusão de acordo com o tipo de classificacao(simples ou multipla)
# e invoca a função de fusão de acordo com o tipo de fusão para a classificação múltipla
def executa_classificacao(idarq, arqs_treino, arqs_clf, arqs_ppi, opt_clfs, tipo='simples', fusao='serie'):    
    resultados = []
    if tipo=='simples':        
        for arq_treino,arq_clf,arq_ppi in zip(arqs_treino, arqs_clf, arqs_ppi):
            logging.info("Arquivo de patches: "+arq_ppi)            
            r = classificacao_arquivo(arq_treino, arq_clf, arq_ppi, opt_clfs[0])
            resultados.append(r)
    else:
        if fusao == 'voto':
           loga_sai("Ainda não implementado!") 
           #r_tst,r_pred = fusao_voto(arqs_treino, arqs_clf, opt_clfs)
        elif fusao == 'soma':
           loga_sai("Ainda não implementado!")
           #r_tst,r_pred = fusao_soma(arqs_treino, arqs_clf, opt_clfs)
        elif fusao == 'produto':
           loga_sai("Ainda não implementado!")
           #r_tst,r_pred = fusao_produto(arqs_treino, arqs_clf, opt_clfs)
        else: 
           loga_sai("Metodo de fusao de classificadores desconhecido!")
    
    return resultados
    
# Processa os resultados de classificação obtidos
def processa_resultados(resultados, arqs_clf, opt_clfs, idarq, tipo="simples"):    
    n_patches = []
    n_descartados = []
    taxas = []
    matrizes = []
    tempos = []   
    
    for r in resultados:
        if r == None:
            logging.info("Resultado nulo! Algo de errado...")
        else:
            taxas.append(r['taxa_clf'])
            tempos.append(r['tempo'])
            n_patches.append(r['total'])
            n_descartados.append(r['descartados'])
            matrizes.append(r['matriz'])             
    
    logging.info("TAXAS :" + str(taxas))     
    logging.info("TEMPOS: " + str(tempos))
    logging.info("PATCHES: " + str(n_patches))
    logging.info("\% DESCARTADO: " + str(n_descartados))
    logging.info("MATRIZES: " + str(matrizes))
        
    # plota grafico de resultados [reconhecimento vs qtd patches]
    #arq_grafico = path.dirname(arqs_clf[0])+"/plt_"+tipo+"_"+"-".join(opt_clfs)+"_rcxqt_"+idarq+".pdf"
    arq_grafico = "plt_"+tipo+"_"+"-".join(opt_clfs)+"_rcxqt_"+idarq+".pdf"
    plota_grafico(n_patches, taxas, arq_grafico, tituloX="Num. Patches", tituloY="Tx. Reconhecimento")    
    
    # plota grafico de resultados [reconhecimento vs descarte]    
    #arq_grafico = path.dirname(arqs_clf[0])+"/plt_"+tipo+"_"+"-".join(opt_clfs)+"_rcxde_"+idarq+".pdf"
    arq_grafico = "plt_"+tipo+"_"+"-".join(opt_clfs)+"_rcxde_"+idarq+".pdf"
    plota_grafico(n_descartados, taxas, arq_grafico, tituloX="% Descarte", tituloY="Tx. Reconhecimento")    
        
    # plota grafico de resultados [qtd_patches vs tempo]
    #arq_grafico = path.dirname(arqs_clf[0])+"/plt_"+tipo+"_"+"-".join(opt_clfs)+"_t_"+idarq+".pdf"
    arq_grafico = "plt_"+tipo+"_"+"-".join(opt_clfs)+"_t_"+idarq+".pdf"
    plota_grafico(n_patches, tempos, arq_grafico, tituloX="Num. Patches", tituloY="Tempo")    
    

# Insere a informação do erro no log do programa e forca saida, encerrando o programa
def loga_sai(erro):
    logging.info(erro)
    sys.exit(erro)


###############################################################################

def main():     
    t0 = time()     
    parser = OptionParser()
    parser.add_option("-T", "--base-treino", dest="base_treino",
                      help="Localização do arquivo da base de treino")
    parser.add_option("-t", "--base-teste", dest="base_teste",
                      help="Localização do arquivo da base de teste")     
    parser.add_option("-C", "--clf", dest="opt_clf",
                      default='svm',
                      help="Lista de classificadores a serem utilizados. [dt, qda, svm, knn]")       
    parser.add_option("-s", "--tamanho", dest="opt_tamanho",                      
                      help="Tamanho do patch quadrado a ser utilizado.")                                   
    parser.add_option("-v", action="store_true", dest="verbose", help="Exibir saida no console.")
    parser.add_option("-l", "--log", dest="opt_log", help="Arquivo de log a ser criado.")

                  
    (options, args) = parser.parse_args()
    
    ## Cria a entrada de log do programa
    if existe_opt(parser, "opt_log"):
       idarq = options.opt_log
    else:
       idarq=datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    
    arq_log = 'patchwiser-'+idarq+'.log'
    
    logging.basicConfig(filename="logs/{0}".format(arq_log), format='%(message)s', level=logging.INFO) 
    
    if options.verbose:
        # Configura para o log ser exibido na tela
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)            
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)   
        logging.getLogger('').addHandler(console)
    
    logging.info("INICIO DO PROGRAMA")   
    
    # verifica se a base de treino e de teste passadas existem
    if existe_opt(parser, "base_treino"):
        if not (path.isfile(options.base_treino)):
            loga_sai("Erro: Caminho da base de treino incorreto ou o arquivo nao existe.")            
    else:
        loga_sai("Base de treino ausente.")
        
    if existe_opt(parser, "base_teste"):
        if not (path.isfile(options.base_teste)):
            loga_sai("Erro: Caminho da base de teste incorreto ou o diretorio nao existe.")            
    else:
        loga_sai("Base para classificacao ausente")        

    arq_treino = options.base_treino    # arquivos das bases de treino
    arq_clf = options.base_teste       # arquivos dos atributos das bases a classificar    
    
    # verifica se o classificador passado é válido
    if not(CLFS[options.opt_clf]):
       loga_sai("Erro: Classificador desconhecido. Valores aceitos: " + str(CLFS.keys))                
    
    logging.info("<<<<<<<< EXECUTA CLASSIFICAÇÃO >>>>>>>>")
    t2 = time()                  
    resultados = []
    resultados = executa_classificacao(arq_treino, arq_clf, options.opt_clf, arq_log, limiar=0.0)
        
    # loga o tempo de execução da extração    
    logging.info("Tempo total de classificação: " + str(round(time()-t2,3)))
    
    # Processa os resultados        
    #logging.info("<<<<<<<< PROCESSA RESULTADOS >>>>>>>>")
    processa_resultados(resultados, arq_clf, options.opt_clf, arq_log)
    
    # Encerramento e contagem de tempo de execucao     
    logging.info("Tempo total do programa: " + str(round(time()-t0,3)))
    logging.info("ENCERRAMENTO DO PROGRAMA")      
    
    

    
   
# Programa principal 
if __name__ == "__main__":	
	main()
