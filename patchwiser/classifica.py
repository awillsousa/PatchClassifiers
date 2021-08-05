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
from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import sys
import numpy as np
import extrator as ex
import matplotlib.pyplot as plt
import multiprocessing as mp
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

###############################################################################

# Plota grafico de resultados 
def plota_grafico(dadosX, dadosY, arquivo="grafico.pdf", titulo="", tituloX="X", tituloY="Y", ):
    
    plt.plot(dadosX, dadosY)    
    # anota os pontos de classificacao
    for x,y in zip(dadosX,dadosY):
        #plt.annotate(r'$('+str(x)+","+str(round(y,2))+')$',
        #plt.annotate(r'$('+str(x)+","+str(round(y,2))+')$',
        #         xy=(x,y), xycoords='data',
        #         xytext=(+10, +30), textcoords='offset points', fontsize=10,
        #         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.1"))

        plt.plot([x,x],[0,y], color ='green', linewidth=.5, linestyle="--")
        plt.plot([x,0],[y,y], color ='green', linewidth=.5, linestyle="--")
        plt.scatter([x,],[y,], 50, color ='red')
        
    plt.ylabel(tituloY)
    plt.xlabel(tituloX)
    if (titulo == ""):
        titulo = tituloX + " vs " + tituloY
    plt.title(titulo)
    
    # Configura limites dos eixos
    plt.xlim(0.0, max(dadosX)+10)
    max_y = max(dadosY)
    if max_y < 100:
        max_y = 100
        
    plt.ylim(0.0, max_y)    
    plt.savefig(arquivo)
    plt.clf()
       

############################################################################### 

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
   
def classifica_img_proba(imagem, clf, atrib_ts):
    logging.info("Classificacao imagem " + imagem)
    
    # recupera o rotulo real da imagem
    classe, _ = ex.classe_arquivo(imagem)                
    rotulo_real = ex.CLASSES[classe]
            
    preds_prob = clf.predict_proba(atrib_ts)     
    probs_img = np.max(preds_prob, axis=0)
        
    ls_preds = np.where(preds_prob[:,0] > preds_prob[:,1], 0, 1)   
    print("ls_preds: {0}".format(ls_preds.shape))    
    rotulo_pred = np.argmax(np.bincount(ls_preds))
    #rotulo_pred = np.argmax(probs_img)
    print("rotulo_pred: {0}".format(rotulo_pred))
    errados = len([x for x in ls_preds if x != rotulo_real])
    
    return (rotulo_real, rotulo_pred, errados, probs_img)   
    
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
def classificacao_arquivo(base_tr, base_ts, arq_ppi, id_clf, idarq=''):
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
        
        r_tst = []  # lista dos rotulos reais das imagens
        r_pred = [] # lista dos rotulos predito das imagens
        probs_imgs = []
        total_erro = 0
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
                tst, pred, erro, prob = classifica_img_proba(imagem['arquivo'], clf, atribs_img)
                r_tst.append(tst)
                r_pred.append(pred)
                total_erro += erro
                probs_imgs.append(prob)
                
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
        
        # Calcula curva ROC/AUC        
        probas_ = np.asarray(probs_imgs)        
        fpr, tpr, thresholds = roc_curve(r_tst.ravel(), probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        
        label = "Acc: {0}".format(round(taxa_clf,2))
        helper.plot_roc(fpr,tpr,roc_auc,label,idarq)        
        
        tempo_exec = time()-inicio        
                         
        total_imgs = len(imagens)
        total_patches = total_imgs*num_ppi                         
        
        # armazena os resultados
        resultado = { 'ppi': num_ppi,               # patches utilizados por imagem
                      'descartados': 0,    # total de patches descartados
                      'total': imagens[0]['total'], # total de patches gerados para a imagem
                      'taxa_clf': round(taxa_clf,3),  # taxa de classificacao 
                      'erro_ptx' :  total_erro/total_patches,
                      'tempo': round(tempo_exec,3),   # tempo de execucao da classificacao
                      'matriz': cm,           # matriz de confusao                      
                      'roc': (fpr,tpr,roc_auc)    # curva ROC
                    }
        
        
        return (resultado)
    except Exception as e:
        logging.info(str(e))


def clf_arquivo_paralelo(base_tr, base_ts, arq_ppi, id_clf, idarq=''):
    inicio = time()    
    imagens = {}
    clf = get_clf(id_clf)    
    logging.info("<<<<<<<< classificacao_arquivo PARALELO>>>>>>>>")    
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
        
        r_tst = []  # lista dos rotulos reais das imagens
        r_pred = [] # lista dos rotulos predito das imagens
        probs_imgs = []
        total_erro = 0
        idx1 = 0    # posicao inicial dos atributos da imagem
        idx2 = 0    # posicao final dos atributos da imagem
        num_ppi = imagens[0]['total']
        total_desc = 0      # total de patches descartados
        
        # Carrega os atributos de acordo com as informações do arquivos de patches por imagem (.ppi)        
        tempos_imgs = []
        vals_clfs = []
        with mp.pool.Pool(20) as p:
            for imagem in imagens:
                t0_imagem = time()            
                idx2 = imagem['ppi']            
                if idx2 > 0:
                    idx2 += idx1  # limite superior da fatia                
                    atribs_img = atrib_ts[idx1:idx2]                 
                    #tst, pred, erro, prob = classifica_img_proba(imagem['arquivo'], clf, atribs_img)
                    proc = p.apply_async(classifica_img_proba, (imagem['arquivo'], clf, atribs_img))
                    vals_clfs.append(proc.get())                    
                    total_desc += imagem['descartados']                
                    idx1 = idx2
                tempos_imgs.append(round(time()-t0_imagem,3))    
                logging.info("Tempo classificação imagem: " + str(tempos_imgs[-1]))        
            
        
        for tst, pred, erro, prob in vals_clfs:
            r_tst.append(tst)
            r_pred.append(pred)
            total_erro += erro
            probs_imgs.append(prob)
        
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
        
        # Calcula curva ROC/AUC        
        probas_ = np.asarray(probs_imgs)        
        fpr, tpr, thresholds = roc_curve(r_tst.ravel(), probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        
        label = "Acc: {0}".format(round(taxa_clf,2))
        helper.plot_roc(fpr,tpr,roc_auc,label,idarq)        
        
        tempo_exec = time()-inicio        
                         
        total_imgs = len(imagens)
        total_patches = total_imgs*num_ppi                         
        
        # armazena os resultados
        resultado = { 'ppi': num_ppi,               # patches utilizados por imagem
                      'descartados': 0,    # total de patches descartados
                      'total': imagens[0]['total'], # total de patches gerados para a imagem
                      'taxa_clf': round(taxa_clf,3),  # taxa de classificacao 
                      'erro_ptx' :  total_erro/total_patches,
                      'tempo': round(tempo_exec,3),   # tempo de execucao da classificacao
                      'matriz': cm,           # matriz de confusao                      
                      'roc': (fpr,tpr,roc_auc)    # curva ROC
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
            #r = classificacao_arquivo(arq_treino, arq_clf, arq_ppi, opt_clfs[0], idarq)
            r = clf_arquivo_paralelo(arq_treino, arq_clf, arq_ppi, opt_clfs[0], idarq)
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
    
'''    
Processa os resultados de classificação obtidos
'''
def processa_resultados(resultados, arqs_clf, opt_clfs, arq_log, tipo="simples"):    
    r_print = {         'ppi': {'exibe':True,  'label': "Patches por Imagem (Usados)", 'valores': []},          # patches utilizados por imagem
                      'total': {'exibe':False, 'label': "Patches por Imagem (Gerados)", 'valores': []},          # patches utilizados por imagem
                'descartados': {'exibe':True,  'label': "%Patches Descartados", 'valores': []},    # total de patches descartados
                   'tam_orig': {'exibe':True,  'label': "Tamanho Orignal Treinamento", 'valores': []},    # tamanho da base de treinamento para cada uma das classes
                    'tam_rdz': {'exibe':True,  'label': "Tamanho Base Reduzida", 'valores': []},    # tamanho da base reduzida obtida
                   'taxa_clf': {'exibe':True,  'label': "Tx Classificacao", 'valores': []},  # taxa de classificacao 
                  'erro_ptx' : {'exibe':True,  'label': "Erro (Nivel de Patch)", 'valores': []},
                      'tempo': {'exibe':True,  'label': "Tempo de Execucao", 'valores': []},   # tempo de execucao da classificacao
                     'matriz': {'exibe':True,  'label': "Matrizes de Confusao", 'valores': []},           # matriz de confusao                      
                        'roc': {'exibe':True,  'label': "Curva ROC", 'valores': []},
                        'limiar': {'exibe':True,  'label': "Limiar", 'valores': []}
              }
            
    for r in resultados:
        if r == None:
            logging.info("Resultado nulo! Algo de errado...")
        else:
            for chave, elem in r.items():                
                r_print[chave]["valores"].append(elem)
    
    for chave,elem in r_print.items():
        if elem['exibe']:
           logging.info("{0}: {1}".format(elem["label"],str(elem["valores"])))     
    
    # Plota as curvas do treinamento
    id_visualiz = path.basename(arq_log).replace(".log","")    
    
    labels = ["Acc: " + str(x) + " Limiar: "+ str(y) for x,y in zip(r_print['taxa_clf']['valores'], r_print['limiar']["valores"])]        
    helper.plot_rocs(r_print['roc']["valores"], labels=labels, id_arquivo=id_visualiz, titulo="Curvas de ROC Limiares")            
         
    

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
    parser.add_option("-f", "--fusao", dest="fusao_clf",
                  help="Método de fusão de classificadores a utilizar. [voto, soma, produto]")    
    parser.add_option("-m", "--metodo", dest="opt_metodo",
                      default='pftas',
                      help="Metodo de extração de atributos. [pftas, lbp, glcm]")   
    parser.add_option("-p", "--prefixo", dest="opt_prefixo",
                      default='base_',
                      help="Prefixo dos arquivos a serem gerados")           
    parser.add_option("-a", "--altura", dest="opt_altura",
                      #default=4,
                      help="Altura da quadtree de patches geradas.")    
    parser.add_option("-n", "--nivel", dest="opt_nivel",                      
                      help="Nível da quadtree de patches a ser utilizado.")  
    parser.add_option("-s", "--tamanho", dest="opt_tamanho",                      
                      help="Tamanho do patch quadrado a ser utilizado.")                           
    parser.add_option("-i", "--incremento", dest="opt_incremento",                      
                      help="Tamanho do incremento a ser utilizado.") 
    parser.add_option("-d", "--descartar", dest="descarte", default=False, 
                      help="Indicar se descarte de patches será usado.")
    parser.add_option("-v", action="store_true", dest="verbose", help="Exibir saida no console.")
    parser.add_option("-l", "--log", dest="opt_log", help="Arquivo de log a ser criado.")

                  
    (options, args) = parser.parse_args()
    
    ## Cria a entrada de log do programa
    if existe_opt(parser, "opt_log"):
       idarq = options.opt_log
    else:
       idarq  = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
       
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
        if not (path.isdir(options.base_treino)):
            loga_sai("Erro: Caminho da base de treino incorreto ou o arquivo nao existe.")            
    else:
        loga_sai("Base de treino ausente.")
        
    if existe_opt(parser, "base_teste"):
        if not (path.isdir(options.base_teste)):
            loga_sai("Erro: Caminho da base de teste incorreto ou o diretorio nao existe.")            
    else:
        loga_sai("Base para classificacao ausente")        
    
    # verifica se será utilizado descarte de patches
    descarta=False    
    if existe_opt(parser, "descarte"):
       descarta=True
       DESCARTA=True
       
    # Seta o valor do incremento dos patches de tamanho fixo    
    inc=ex.INCR_PATCH   
    if existe_opt(parser, "opt_incremento"):
       try: 
           inc = int(options.opt_incremento)    
       except ValueError:
           loga_sai("Valor de incremento inválido! Um inteiro era esperado.")
                          
      
    # metodo de extracao de atributos a ser utilizado
    if not(existe_opt(parser, "opt_metodo")):
       loga_sai("Erro: Metodo de extração ausente.")       
        
    # verifica o tipo de divisão de patches que sera utilizada    
    fl_altura = existe_opt(parser, "opt_altura")
    fl_nivel = existe_opt(parser, "opt_nivel")
    fl_tamanho = existe_opt(parser, "opt_tamanho")
    arqs_treino = []    # arquivos das bases de treino
    arqs_clf = []       # arquivos dos atributos das bases a classificar
    # pelo menos um tipo de divisao deve ser escolhido
    if not(fl_altura) and not(fl_nivel) and not(fl_tamanho):
       loga_sai("Erro: Pelo menos uma divisão dos patches deve ser escolhida entre tamanho, nivel da quadtree ou altura da quadtree de patches.")
    else:
        # apenas uma das opcoes deve ser passada
        if not(helper.xor_n(fl_altura,fl_nivel,fl_tamanho)):
           loga_sai("Erro: Apenas uma opção de divisão dos patches deve ser escolhida.")         
        else: # somente um tipo de divisão
           try: 
               logging.info("<<<<<<<< EXECUTA EXTRAÇÃO >>>>>>>>")
               if fl_altura:  # altura (da quadtree) foi passada                                       
                   altura = int(options.opt_altura)                   
                   logging.info("Executando extração para quadtree: altura = {0}".format(altura))
                   arqs_treino, arqs_ppi_tr = extrai.executa_extracao(options.base_treino, options.opt_metodo, "ALTURA", options.opt_prefixo, descarta, altura)
                   logging.info("Tempo total de extracao do treinamento: " + str(round(time()-t0,3)))
                   t1 = time()
                   arqs_clf, arqs_ppi_ts = extrai.executa_extracao(options.base_teste, options.opt_metodo, "ALTURA", options.opt_prefixo, descarta, altura)
                   logging.info("Tempo total de extracao do teste: " + str(round(time()-t1,3)))
                   
               elif fl_nivel:   # nivel foi passado
                   nivel = int(options.opt_nivel)
                   logging.info("Executando extração para quadtree: nivel = {0}".format(nivel))
                   arqs_treino, arqs_ppi_tr = extrai.executa_extracao(options.base_treino, options.opt_metodo, "NIVEL", options.opt_prefixo, descarta, nivel)                   
                   logging.info("Tempo total de extracao do treinamento: " + str(round(time()-t0,3)))
                   t1 = time()
                   arqs_clf, arqs_ppi_ts = extrai.executa_extracao(options.base_teste, options.opt_metodo, "NIVEL", options.opt_prefixo, descarta, nivel)        
                   logging.info("Tempo total de extracao do teste: " + str(round(time()-t1,3)))
               else:  # tamanho foi passado
                   opt_tams = options.opt_tamanho.split(',')
                   
                   if (len(opt_tams) > 1): # passado um intervalo                           
                      logging.info("Executando extração para intervalo de tamanhos fixos: de {0}x{0} a {1}x{1} com incrementos de {2}".format(int(opt_tams[0]), int(opt_tams[1]), inc))                          
                      arqs_treino, arqs_ppi_tr = extrai.executa_extracao(options.base_treino, options.opt_metodo, "TAMANHO", options.opt_prefixo, descarta, int(opt_tams[0]), int(opt_tams[1]), inc)
                      logging.info("Tempo total de extracao do treinamento: " + str(round(time()-t0,3)))
                      t1 = time()
                      arqs_clf, arqs_ppi_ts = extrai.executa_extracao(options.base_teste, options.opt_metodo, "TAMANHO", options.opt_prefixo, descarta, int(opt_tams[0]), int(opt_tams[1]), inc)    
                      logging.info("Tempo total de extracao do teste: " + str(round(time()-t1,3)))
                   else: # passado apenas um tamanho
                      logging.info("Executando extração para tamanho fixo: {0}x{0}".format(int(opt_tams[0])))     
                      arqs_treino, arqs_ppi_tr = extrai.executa_extracao(options.base_treino, options.opt_metodo, "TAMANHO", options.opt_prefixo, descarta, int(opt_tams[0]), int(opt_tams[0]), inc)
                      logging.info("Tempo total de extracao do treinamento: " + str(round(time()-t0,3)))
                      t1 = time()
                      arqs_clf, arqs_ppi_ts = extrai.executa_extracao(options.base_teste, options.opt_metodo, "TAMANHO", options.opt_prefixo, descarta, int(opt_tams[0]), int(opt_tams[0]), inc)
                      logging.info("Tempo total de extracao do teste: " + str(round(time()-t1,3)))
           except ValueError as v:                
                loga_sai("Valor inválido, quando um inteiro era esperado. Erro: " + str(v))            
    
    if not arqs_treino:
        loga_sai("Erro ao recuperar nome dos arquivos de treino!")
    
    # loga o tempo de execução da extração
    logging.info("Tempo total de extracao: " + str(round(time()-t0,3)))
    
    # verifica as listas de arquivos            
    if len(arqs_treino) != len(arqs_clf):
       loga_sai("Divergencia entre arquivos da base de treinamento e de classificação.") 
    
    ##print("Arquivos de treino: "+str(arqs_treino))    #apenas para debugar
    ##print("Arquivos de classificação: "+str(arqs_clf))   #apenas para debugar
    ##print("Arquivos ppi: " + str(arqs_ppi_ts))    #apenas para debugar
    
    if existe_opt(parser, "opt_clf"):       
        opt_clfs = options.opt_clf.split(',')     
    else:
        ### executar apenas a extracao  GO HORSE!  ###                    
        loga_sai("Executados apenas os procedimentos de extração...") 
        
        
    # passado mais de um classificador sem um metodo de fusao definido    
    if (len(opt_clfs) > 1):
        if not(existe_opt(parser,'fusao_clf')):
            loga_sai("Erro: Metodo de fusao de classificadores ausente.")
        # verifica se os metodo de fusao definido é válido
        if not(options.fusao_clf in FUSAO):
           loga_sai("Erro: Metodo de fusao desconhecido. Valores aceitos: " + str(FUSAO)) 
        
    # verifica se os classificadores passados são validos
    for c in opt_clfs:
        if not(CLFS[c]):
           loga_sai("Erro: Classificador desconhecido. Valores aceitos: " + str(CLFS.keys))                
    
    '''
     Processo de extracao de atributos concluida
     Execucao dos processos de classificacao
    '''
    
    logging.info("<<<<<<<< EXECUTA CLASSIFICAÇÃO >>>>>>>>")
    t2 = time()  
    # Execução da classificação
    if (len(opt_clfs) > 1): # ha mais de uma classificador para fundir
        tipo = "multipla"        
        logging.info("Classificacao MULTIPLA")
        logging.info("Opções de classificação: {0}".format("-".join(opt_clfs)))            
        resultados = executa_classificacao(idarq, arqs_treino, arqs_clf, arqs_ppi_ts, opt_clfs, tipo, options.fusao_clf)       
    else: # apenas um classificador
        tipo = "simples" 
        logging.info("Classificacao SIMPLES")
        logging.info("Opções de classificação: {0}".format("-".join(opt_clfs)))          
        resultados = executa_classificacao(idarq, arqs_treino, arqs_clf, arqs_ppi_ts, opt_clfs)    
        
    # loga o tempo de execução da extração    
    logging.info("Tempo total de classificação: " + str(round(time()-t2,3)))
    
    # Processa os resultados        
    logging.info("<<<<<<<< PROCESSA RESULTADOS >>>>>>>>")
    processa_resultados(resultados, arqs_clf, opt_clfs, idarq)
    
    
    # Encerramento e contagem de tempo de execucao     
    logging.info("Tempo total do programa: " + str(round(time()-t0,3)))
    logging.info("ENCERRAMENTO DO PROGRAMA")     
    
    

    
   
# Programa principal 
if __name__ == "__main__":	
	main()
