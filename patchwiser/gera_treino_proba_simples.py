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
from scipy.sparse import csr_matrix, hstack,vstack
from scipy import interp
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score,roc_curve, auc
import sys
import numpy as np
import extrator as ex
import matplotlib.pyplot as plt
import helper
import logging
import multiprocessing as mp 
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
Classifica uma imagem utilizando o resultado de dois classificadores. O primeiro (clf) foi treinado
com instâncias que foram corretamente identificadas durante o processo de treinamento utilizando 
k-folds. O segundo (clf_rej) foi treinado utilizando instâncias que foram classificadas incorretamente
durante o processo de treinamento. 
A probabilidade final da imagem é dada pelo produto da predição do primeiro + (1 - predicao do segundo)

'''
def classifica_img_proba(imagem, clf, atrib_ts):
    logging.info("Classificacao imagem " + imagem)
    
    # recupera o rotulo real da imagem
    classe, _ = ex.classe_arquivo(imagem)                
    rotulo_real = ex.CLASSES[classe]
            
    preds_prob = clf.predict_proba(atrib_ts) 
    print("preds_prob: {0}".format(preds_prob.shape))
    probs_img = np.max(preds_prob, axis=0)
    print("probs_img: {0}".format(probs_img.shape))
    
    ls_preds = np.where(preds_prob[:,0] > preds_prob[:,1], 0, 1)   
    print("ls_preds: {0}".format(ls_preds.shape))    
    #rotulo_pred = np.argmax(np.bincount(ls_preds))
    rotulo_pred = np.argmax(probs_img)
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
Classifica uma base de imagens utilizando patches para isso, além de basear-se nos valores de
probabilidade preditos. 
'''
def classificacao_probas(atrib_tr, rotulos_tr, base_ts, arq_ppi, id_clf):
    inicio = time()    
    imagens = {}
    clf = SVC(gamma=0.5, C=32, cache_size=500, probability=True)
    #clf = get_clf(id_clf)
    
    logging.info("<<<<<<<< classificacao_base >>>>>>>>")    
    try: 
        # Carrega a base de treinamento      
        if atrib_tr == None:
            loga_sai("Falha na carga da base de treinamento" )        
    
        # Treina o classificador
        logging.info("Treinando classificador...")
        clf.fit(atrib_tr, rotulos_tr)                    
        
                
        # Carrega a base de testes e o arquivo de patches por imagem                             
        atrib_ts, rotulos_ts = load_svmlight_file(base_ts, dtype=np.float32, n_features=162)
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
        #total_desc = 0      # total de patches descartados
        
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
                #total_desc += imagem['descartados']                
                idx1 = idx2
            tempos_imgs.append(round(time()-t0_imagem,3))    
            logging.info("Tempo classificação imagem: " + str(tempos_imgs[-1]))
            
        # Loga estatisticas de tempo por imagem
        logging.info("Tempo medio de classificacao por imagem: {0}".format(np.mean(tempos_imgs)))
        logging.info("Desvio padrao tempo classificacao por imagem: {0}".format(np.std(tempos_imgs)))
        
        # cria as matrizes de confusao
        cm = confusion_matrix(r_tst, r_pred)
        
        # exibe a taxa de classificacao
        total_imgs = len(imagens)
        total_patches = total_imgs*num_ppi
        
        r_pred = np.asarray(r_pred)
        r_tst = np.asarray(r_tst)
        taxa_clf = np.mean(r_pred.ravel() == r_tst.ravel()) * 100
        logging.info("Taxa de Classificação: %f " % (round(taxa_clf,3)))     
        
        # Calcula curva ROC/AUC        
        probas_ = np.asarray(probs_imgs)        
        fpr, tpr, thresholds = roc_curve(r_tst.ravel(), probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        
        tempo_exec = time()-inicio        
        
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
Exclui linhas repetidas de um array
'''
def unique_linhas(data):    
    s_idx = np.lexsort(data.T)  
    s_data =  data[s_idx,:]
    
    # Mascara de linhas unicas
    mascara = np.append([True],np.any(np.diff(s_data,axis=0),axis=1))
    
    # Recupera as linhas unicas
    return (s_data[mascara])

'''
Agrega as bases obtidas durante o processo de treinamento para formar uma unica base
'''
def agrega_base(atributos, rotulos):
    lista_patches = None
    lista_rotulos = None            
    for atribs,rots in zip(atributos,rotulos):
        if lista_patches == None:
            lista_patches = atribs
            lista_rotulos = rots
        else:
            logging.info("Pos stack lista_patches: {0}  lista_rotulos: {1}".format(lista_patches.shape, lista_rotulos.shape))            
            lista_patches = vstack((lista_patches,atribs))
            lista_rotulos = hstack((lista_rotulos,rots))
    
    lista_patches = np.array(lista_patches.todense())
    lista_rotulos = np.array(lista_rotulos.todense())
    lista_rotulos = lista_rotulos.flatten()
    logging.info("Pos conversao para ndarray lista_patches: {0} lista_rotulos: {1}".format(lista_patches.shape, lista_rotulos.shape))
    
    s_idx = np.lexsort(lista_patches.T)  
    s_data =  lista_patches[s_idx,:]        
    mascara = np.append([True],np.any(np.diff(s_data,axis=0),axis=1))  # Mascara de linhas unicas
    atrib_red = lista_patches[mascara]       # base de atributos reduzida
    rotulos_red = lista_rotulos[mascara]      # lista de rotulos reduzida
    
    return (atrib_red, rotulos_red)

'''
Divide a base de atributos passada em K-folds e seleciona, durante o processo, os patches
que forem classificados com probabilidade maior que o limiar passado em um grupo de patches bons
e os demais em um grupo de patches ruins para classificação
'''
def seleciona_patches(atrib_tr, rotulos_tr, clf, limiar):
    melhores = []
    rots_melhor = []    
    rocs = []
    
    # Geracao k-folds e obtenção da base de treino reduzida
    skf = StratifiedKFold(n_splits=5)
    for tr_idx, ts_idx in skf.split(atrib_tr, rotulos_tr):
        X_tr, X_ts = atrib_tr[tr_idx], atrib_tr[ts_idx]
        y_tr, y_ts = rotulos_tr[tr_idx], rotulos_tr[ts_idx]
        
        probs_patches = clf.fit(X_tr, y_tr).predict_proba(X_ts)
        logging.info("Predicao de probabilidades feita")
        max_probs = np.array([max(float(x[0]),float(x[1])) for x in probs_patches])
        probs_patches = np.array(probs_patches)
        preds = np.where(probs_patches[:,0] > probs_patches[:,1], 0, 1)
                        
        idxs1 = np.where(max_probs > limiar, True,False)
        idxs2 = np.where(preds == y_ts, True, False)
        idxs = np.logical_and(idxs1,idxs2)
        
        # Seleciona os melhores patches baseado no vetor de indices gerado
        print("Lista de indices: {0}".format(len(idxs)))
        melhores.append(X_ts[idxs,:])
        rots_melhor.append(y_ts[idxs])
        logging.info("PRE k-fold melhores: {0}  rotulos: {1}".format(melhores[-1].shape, rots_melhor[-1].shape))
        
        # Adiciona na lista de curvas rocs do treinamento
        # Calcula curva ROC/AUC                
        fpr, tpr, thresholds = roc_curve(y_ts, probs_patches[:, 1])
        rocs.append((fpr,tpr,auc(fpr, tpr)))
        
    return (melhores, rots_melhor, rocs)

'''
Executa o processo de classificação
''' 
def executa_classificacao(base_tr, base_ts, opt_clf, arq_log, limiar=0.9): 
    #resultados = []  
    melhores = []       # lista dos patches classificados com maior confianca (mais discriminativos)
    rots_melhor = []    # lista dos rotulos dos melhores patches
    
    # arquivos de informacoes de patches 
    arq_ppi_tr = base_tr.replace(".svm",".ppi")
    arq_ppi_ts = base_ts.replace(".svm",".ppi")
                         
    logging.info("Arquivo de patches (treino): {0}".format(arq_ppi_tr))            
    logging.info("Arquivo de patches (teste): {0}".format(arq_ppi_ts))            
    
    ## Carga das bases    
    atrib_tr = None 
    rotulos_tr = None             
    atrib_tr, rotulos_tr = load_svmlight_file(base_tr, dtype=np.float32,n_features=162) 
    base = {'data':atrib_tr, 'labels': rotulos_tr}
    # Gera a visualizacao da base de dados antes da reducao
    id_visualiz = path.basename(arq_log).replace(".log","")
    
    if not path.isfile("plt-BHtSNE-{0}.png".format(id_visualiz)):
        helper.visualiza_bhtsne(base, id_visualiz)
    
    id_visualiz = path.basename(arq_log).replace(".log","")+"limiar{0}".format(limiar)    
    
    logging.info("Carregada a base de treinamento: " + base_tr)
    logging.info("Tipo Atributos {0}".format(type(atrib_tr)))
    logging.info("Tipo Rotulos {0}".format(type(rotulos_tr)))
    logging.info("Atributos Shape {0}".format(atrib_tr.shape))
    logging.info("Rotulos Shape {0}".format(rotulos_tr.shape))
    
    clf = get_clf(opt_clf)
    melhores, rots_melhor, rocs = seleciona_patches(atrib_tr, rotulos_tr, clf, limiar)
    atrib_red, rotulos_red = agrega_base(melhores, rots_melhor)
    tam_orig = rotulos_tr.shape[0]
    tam_red = rotulos_red.shape[0]
    reducao=round(100*(tam_orig-tam_red)/tam_orig,2)
    logging.info("Redução da base de treinamento: {0}\%".format(reducao))
    
    # Plota as curvas do treinamento
    #helper.plot_rocs(rocs, ['kfold1','kfold2','kfold3','kfold4','kfold5'], id_arquivo=id_visualiz+"TR", titulo="Curvas de Treino")
                 
    r = classificacao_probas(csr_matrix(atrib_red), rotulos_red, base_ts, arq_ppi_ts, opt_clf)
    r['limiar'] = limiar
    r['descartados'] = reducao
    r['tam_orig'] = str(np.bincount(np.where(rotulos_tr > 0.0, 1, 0)))
    r['tam_rdz'] = str(np.bincount(np.where(rotulos_red > 0.0, 1, 0)))
    
    
    # Gera visualizacao das bases de patches reduzida    
    base_rdz = {"data": atrib_red, "labels": rotulos_red}
    
    # Retirado temporariamente, pois demora muito para calcular
    #if not path.isfile("plt-BHtSNE-{0}.png".format(id_visualiz+"-rdz")):    
    #    helper.visualiza_bhtsne(base_rdz, id_visualiz+"-rdz")
    
    # Todas as curvas serao plotadas juntas no processa_resultados
    # Plota curva ROC
    #helper.plot_roc(r['fpr'],r['tpr'],r['auc'],id_visualiz)
    
    return r

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

    
'''
Insere a informação do erro no log do programa e forca saida, encerrando o programa
'''
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
    parser.add_option("-p", "--prefixo", dest="opt_prefixo",
                      default='base_',
                      help="Prefixo dos arquivos a serem gerados")                   
    parser.add_option("-s", "--tamanho", dest="opt_tamanho",                      
                      help="Tamanho do patch quadrado a ser utilizado.")                               
    parser.add_option("-v", action="store_true", dest="verbose", help="Exibir saida no console.")
    parser.add_option("-l", "--log", dest="opt_log", help="Arquivo de log a ser criado.")
                  
    (options, args) = parser.parse_args()
    
    ## Cria a entrada de log do programa
    if existe_opt(parser, "opt_log"):
       idarq = options.opt_log
    else:
       idarq = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
       
    arq_log = 'tr-proba-'+idarq+'.log'
    
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
    with mp.pool.Pool() as p:
        for l in range(69, 95, 5):
            limiar=round(l/100,2)
            proc = p.apply_async(executa_classificacao, (arq_treino, arq_clf, options.opt_clf, arq_log, limiar))
            resultados.append(proc.get())
            #resultados.append(executa_classificacao(arq_treino, arq_clf, options.opt_clf, arq_log, limiar=round(l/100,2)))    
        
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
