# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 23:21:04 2016

@author: willian

Adaptado dos códigos disponíveis em https://github.com/DEAP/notebooks

Execução do programa:

- Recebe uma base de atritubos (PFTAS) já extraidos, de três (03) diretórios diferentes:
     treino - validacao - teste
- Utiliza um algoritmo de PSO para buscar os melhores valores de atributos Haralick extraidos 
  anteriormente em conjunto com o PFTAS. 
  O PSO realiza a busca por valores que permitam filtrar a base de treinamento. Essa base
  filtrada/reduzida será utilizada ao final do total de gerações do algoritmo genético para
  classificar uma base de testes. 
- A função de fitness para avaliar a qualidade da particula, será o resultado de classificação
  obtido. O resultado será determinado pela AUC da curva ROC obtida durante o processo de treinamento
  utilizando o esquema de k-folds.  
- Treina um classificador com essa base e aplica sobre a base de testes e coleta os resultados, gravando
  a base final obtida. 
"""

import numpy as np
import matplotlib.pyplot as plt
import classifica
import logging
import helper
import extrator as ex
from os import path
from scipy import interp
from sklearn import svm
from sklearn.metrics import roc_auc_score,roc_curve, auc
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize, MinMaxScaler

from sklearn.metrics import confusion_matrix
from deap import creator, base, tools
from time import time
from datetime import datetime
from optparse import OptionParser
from helper import loga_sai, existe_opt

# Qtd de folds a considerar ao avaliar a base filtrada por uma particula
# do PSO
K_FOLDS = 3

'''
Descritores GLCM(Haralick)
haralick_labels = 
["0 - Angular Second Moment (Uniformity) Energy = (Uniformity)^1/2",
 "1 - Contrast",
 "2 - Correlation",
 "3 - Sum of Squares: Variance (Contrast)",
 "4 - Inverse Difference Moment (Local Homogeneity)",
 "5 - Sum Average",
 "6 - Sum Variance",
 "7 - Sum Entropy",
 "8 - Entropy",
 "9 - Difference Variance",
 "10 - Difference Entropy",
 "11 - Information Measure of Correlation 1",
 "12 - Information Measure of Correlation 2",
 "13 - Maximal Correlation Coefficient"]
'''     
IDXS_GLCM = [0,1,2,4,5,8]
IDX_FILTRO = [2, 8]
M_BASE = []

# Lista de posicoes do espaco de busca que ja foram visitadas por alguma
# particula do PSO. Essa lista sera usada para evitar que o mesmo local do
# espaco de busca seja visitado mais de uma vez por diferentes particulas
VISITADOS = []

creator.create("FitnessAUC", base.Fitness, weights=(1.0,-1.0))
creator.create("Particle", np.ndarray, fitness=creator.FitnessAUC, speed=list, smin=None, smax=None, best=None, idxbase=None)

'''
Carrega uma base a partir de um arquivo .svm
'''
def carrega_base(arq_base, n_features=162):
    atribs = None 
    rotulos = None              
    atribs, rotulos, qid = load_svmlight_file(arq_base, n_features=n_features, query_id=True)#dtype=np.float32 
    
    return (atribs, rotulos, qid)
    
'''
Gera valores de particula    
'''
def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(np.random.uniform(pmin, pmax, size)) 
    part.speed = np.random.uniform(smin, smax, size)
    part.smin = smin
    part.smax = smax
    part.pmin = pmin
    part.pmax = pmax
    
    return part

    
'''
Atualiza os valores das particulas
'''    
def updateParticle(part, best, phi1, phi2):
    u1 = np.random.uniform(0, phi1, len(part))
    u2 = np.random.uniform(0, phi2, len(part))
    v_u1 = u1 * (part.best - part)
    v_u2 = u2 * (best - part)
    part.speed += v_u1 + v_u2
    for i, speed in enumerate(part.speed):
        if speed < part.smin:
            part.speed[i] = part.smin
        elif speed > part.smax:
            part.speed[i] = part.smax
    part += part.speed
    
    for i, pos in enumerate(part):
        pos = round(pos,4)
        if pos < part.pmin:
            part[i] = part.pmin
        elif pos > part.pmax:
            part[i] = part.pmax
    
    
'''
Filtra uma base de atributos passada e a lista dos seus respectivos rotulos,
de acordo com os valores de filtros passados. Os valores de filtro são aplicados
sobre os valores de atributos glcm (base_glcm) e a partir dela os indices de seleção 
são gerados para selecionar a 
'''    
def filtra_base(base_tr, rotulos_tr, atrs_glcm, filtro):    
    # Verificações de tamanho das matrizes    
    if base_tr.shape[0] == 0 or rotulos_tr.shape[0] == 0:
        loga_sai("A base a ser filtrada não pode ser vazia")
    elif not (base_tr.shape[0] == rotulos_tr.shape[0]):
        loga_sai("Quantidade de instâncias e rótulos diferem")
    
    # Filtra a base passada            
    idxs = []
    linhas = atrs_glcm.shape[0]
    conta_true = 0
    for l in range(linhas):         
        condicao = atrs_glcm[l][IDX_FILTRO[0]] < filtro[0] and atrs_glcm[l][IDX_FILTRO[1]] < filtro[1]
        idxs.append(condicao)

        if condicao == True:
            conta_true += 1
                
    idxs = np.where(idxs)[0]
    base_filt = base_tr[idxs, :]
    rotulos_filt = rotulos_tr[idxs]
    
    '''
    #debug
    if (conta_true > 0):
        print("Total de indices: " + str(conta_true))
        print("Shape indices: " + str(idxs.shape))
        print("Shape base filtrada: " + str(base_filt.shape))
        print("Shape rotulos filtrados: " + str(rotulos_filt.shape)+"\n")
    #debug
    '''
    return (base_filt, rotulos_filt, idxs)
    
'''
Avalia a taxa de classificacao obtida a partir da base de treinamento filtrada, 
utilizando os parâmetros da particula para realizar a filtragem da base.
Os valores da particulas representam valores dos atributos de Haralick extraídos
anteriormente para cada patch da base.
Os valores utilizados para filtragem serão (em ordem): ["Entropy", "Correlation"]                   
'''
def avalia_particula(particula):
    #logging.info("Particula: {0}".format(str(particula)))
    
    base_tr = BASE_TR
    rotulos_tr = ROTULOS_TR    
    base_glcm = BASE_GLCM
    
    # Atualiza lista de visitados
    VISITADOS.append((particula[0], particula[1]))
    
    # Filtra a base baseado nos valores da particula
    base_uso, rotulos_uso, idx_uso = filtra_base(base_tr, rotulos_tr, base_glcm, filtro=particula)    
    
    if rotulos_uso.shape[0] < 0.15*rotulos_tr.shape[0]:
       logging.info("Base filtrada muito pequena: {0} linhas.".format(rotulos_uso.shape[0]))  
       return (0.0, 99999.0)
        
    # Faz a divisão da base de treino em folds
    skf = StratifiedKFold(n_splits=K_FOLDS)    
             
    clf = classifica.get_clf("rf")
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    
    for tr_idx, ts_idx in skf.split(base_uso, rotulos_uso):
        X_train, X_test = base_uso[tr_idx], base_uso[ts_idx]
        y_train, y_test = rotulos_uso[tr_idx], rotulos_uso[ts_idx]
        
        # Compute ROC curve and area the curve
        probas_ = clf.fit(X_train, y_train).predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])            
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
            
    mean_tpr /= K_FOLDS
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)                    
            
    particula.idxbase = idx_uso
    
    return (round(mean_auc,4), rotulos_uso.shape[0])    
    
def classifica_img_proba(imagem, clf, atrib_ts):
    logging.info("Classificacao imagem " + imagem)
    
    # recupera o rotulo real da imagem
    classe, _ = ex.classe_arquivo(imagem)   
    rotulo_real = ex.CLASSES[classe]
            
    pesos = clf.predict_proba(atrib_ts)
    probs_img = np.multiply(pesos[:,0], pesos[:,1])
    ls_preds = np.where(pesos[:,0] > pesos[:,1], 0, 1)
    rotulo_pred = np.argmax(np.bincount(ls_preds))
    
    errados = len([x for x in ls_preds if x != rotulo_real])
    
    return (rotulo_real, rotulo_pred, errados, probs_img)


'''
Classifica uma base de imagens utilizando patches para isso, além de basear-se nos valores de
probabilidade preditos. 
'''
def classificacao_probas(atrib_tr, rotulos_tr, base_ts, arq_ppi, id_clf):
    inicio = time()    
    imagens = {}
    clf = svm.SVC(gamma=0.5, C=32, cache_size=500, probability=True)   
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
        imagens = classifica.dicionario_imgs(helper.load_csv(arq_ppi))
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
        #print(str(probas_.shape))
        fpr, tpr, thresholds = roc_curve(r_tst.ravel(), probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        
        tempo_exec = time()-inicio        
        
        # armazena os resultados
        resultado = { 'ppi':num_ppi,               # patches utilizados por imagem
                      'descartados':0,    # total de patches descartados
                      'total':imagens[0]['total'], # total de patches gerados para a imagem
                      'taxa_clf':round(taxa_clf,3),  # taxa de classificacao 
                      'erro_ptx' :  total_erro/total_patches,
                      'tempo':round(tempo_exec,3),   # tempo de execucao da classificacao
                      'matriz':cm,           # matriz de confusao                      
                      'tpr': tpr,
                      'fpr': fpr,
                      'auc': roc_auc
                    }
        
        return (resultado)
    except Exception as e:
        logging.info(str(e))
    
'''    
Processa os resultados de classificação obtidos
'''
def processa_resultados(resultados):    
    r_print = { 'ppi': {'exibe':True, 'label': "Patches por Imagem (Usados)", 'valores': []},          # patches utilizados por imagem
                'total': {'exibe':False, 'label': "Patches por Imagem (Gerados)", 'valores': []},          # patches utilizados por imagem
                'descartados': {'exibe':True, 'label': "%Patches Descartados", 'valores': []},    # total de patches descartados
                'taxa_clf': {'exibe':True, 'label': "Tx Classificacao", 'valores': []},  # taxa de classificacao 
                'erro_ptx' :  {'exibe':True, 'label': "Erro (Nivel de Patch)", 'valores': []},
                'tempo':  {'exibe':True, 'label': "Tempo de Execucao", 'valores': []},   # tempo de execucao da classificacao
                'matriz':  {'exibe':True, 'label': "Matrizes de Confusao", 'valores': []},           # matriz de confusao                      
                'tpr':  {'exibe':False, 'label': "TPRs", 'valores': []},
                'fpr':  {'exibe':False, 'label': "FPRs", 'valores': []},
                'auc':  {'exibe':False, 'label': "AUCs", 'valores': []}
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



# PROGRAMA PRINCIPAL    
###################################################################################################################

if __name__ == "__main__":
    toolbox = base.Toolbox()
    toolbox.register("particle", generate, size=2, pmin=0.001, pmax=0.9999, smin=-1, smax=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", updateParticle, phi1=0.3, phi2=0.8)
    toolbox.register("evaluate", avalia_particula)

    t0 = time()    
    ### CARREGA OPCOES PASSADAS     
    parser = OptionParser()
    parser.add_option("-T", "--base-treino", dest="base_treino",
                      help="Localização do arquivo da base de treino")
    parser.add_option("-t", "--base-teste", dest="base_teste",
                      help="Localização do arquivo da base de teste") 
    parser.add_option("-p", "--populacao", dest="n_pop",
                      help="Tamanho da populacao de particulas") 
    parser.add_option("-g", "--geracoes", dest="n_ger",
                      help="Numero de geracoes do PSO") 
    parser.add_option("-v", action="store_true", dest="verbose", help="Exibir saida no console.")
    parser.add_option("-l", "--log", dest="opt_log", help="Arquivo de log a ser criado.")
    
                      
    (options, args) = parser.parse_args()   
    n_pop = int(options.n_pop)
    n_ger = int(options.n_ger)     
    
    if not n_pop:
       n_pop = 10
    if not n_ger:
       n_ger = 5	

    ## Cria a entrada de log do programa
    if existe_opt(parser, "opt_log"):
       idarq = options.opt_log
    else:
       idarq=datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')       
    
    arq_log='genpatch-'+idarq+'.log'
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
    
    ### CONFIGURACOES INICIAIS E CARGA DAS BASES    
    # Carrega base de treino
    arq_base_tr = options.base_treino     
    arq_base_glcm = arq_base_tr.replace(".svm",".glcm")           
    
    logging.info("Base de Treino: " + arq_base_tr)    
    BASE_TR, ROTULOS_TR, QID = carrega_base(arq_base_tr)    
    BASE_GLCM, _,_ = carrega_base(arq_base_glcm,n_features=14)   
    
    BASE_GLCM = BASE_GLCM.toarray()
    BASE_GLCM = MinMaxScaler().fit_transform(BASE_GLCM)
    
    # carrega base de testes
    arq_base_ts = options.base_teste
    
    pop = toolbox.population(n=n_pop)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    
    '''
    fitnesses = map(toolbox.evaluate, pop)
    for part, fit in zip(pop, fitnesses):
        part.fitness.values = fit
        ##print("Particula: {0} - AUC: {1}".format(str(part), str(part.fitness.values)))
    '''        
        
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields
                     
    GEN = n_ger
    global_best = None
    melhores = []
    for g in range(GEN):        
        logging.info("Geracao {0}".format(str(g)))    
        # avalia a melhor particula
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            
            logging.info("Particula: {0} - Fitness: {1}".format([str(round(v,4)) for v in part], [str(round(f,4)) for f in part.fitness.values]))
            
            # fitness atual é melhor que o melhor resultado ?
            if part.best is None or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
                part.best.idxbase = part.idxbase
            
            # particula atual é melhor que o melhor global ?
            if global_best is None or global_best.fitness < part.fitness:
                global_best = creator.Particle(part)
                global_best.fitness.values = part.fitness.values
                global_best.idxbase = part.idxbase
                melhores.append(global_best)
        logging.info("Geracao {0}: Melhor AUC {1} - Particula {2}".format(g, [str(round(v,4)) for v in global_best.fitness.values], [str(round(p,4)) for p in global_best]))
            
        # atualiza as particulas
        for part in pop:            
            toolbox.update(part, global_best)
            
            # vá buscar de lugares desconhecidos
            tenta = 0
            while (part[0], part[1]) in VISITADOS:
                print("Nova Posicao já visitada - Gerando uma nova...")
                toolbox.update(part, global_best)
                tenta += 1
                if tenta > 10:
                    break
        
        # Junta todos os valores de fitness em uma unica lista e exibe os resultados
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        logging.info(logbook.stream)
            
    logging.info("Tempo total de seleção do AG: %0.2f" % (round(time()-t0,2)))
    
    if (global_best.idxbase == None):
        loga_sai("PSO falhou em obter uma base reduzida! Tente com outros valores.")
    
    #melhores = sorted(melhores, key=lambda k:(k[1], k[0]), reverse=False)        
    #global_best = melhores[0]
    #logging.info("Melhores particulas: {0}".format(str(melhores)))
    logging.info("Melhor Particula: {0} - Fitness: {1}".format([str(round(g,4)) for g in global_best], [str(round(v,4)) for v in global_best.fitness.values]))
               
    # APLICA A MELHOR BASE REDUZIDA NA BASE DE TESTE
    logging.info("Base de Teste: " + arq_base_ts)    
    t2 = time()
            
    BEST_BASE = BASE_TR[global_best.idxbase, :]       # carrega a melhor base obtida pelo PSO
    BEST_ROTS = ROTULOS_TR[global_best.idxbase]    # carrega os rotulos da melhor base do PSO
    #BEST_IDS  = QID[global_best.idxbase]     
    
    clf = classifica.get_clf("svm") 

    # Treina um classificador usando apenas os exemplares definidos pela pela melhor particula    
    r = classificacao_probas(BEST_BASE, BEST_ROTS, arq_base_ts, arq_base_ts.replace(".svm",".ppi"), "svm")    
    
    processa_resultados([r])    
    logging.info("Taxa de Classificação: %0.2f " % (r['taxa_clf']))    
    logging.info("Tempo de classificação: %0.2f" % (time()-t2))
    tam_orig = ROTULOS_TR.shape[0]
    tam_red = BEST_ROTS.shape[0]
    reducao = round(100*(tam_orig-tam_red)/tam_orig,2)    
    logging.info("Taxa de Redução obtida: {0}".format(reducao))
    
    # Armazena a base obtida
    base_genpatch = "bases/"+arq_log.replace(".log",".svm")
                                
    dump_svmlight_file(BEST_BASE, BEST_ROTS, base_genpatch)   
    
    # Plota curva ROC
    helper.plot_roc(r['fpr'],r['tpr'],r['auc'], arq_log.replace(".log", ""))
        
    # Visualiza a base obtida
    ##base_rdz = {"data":BEST_BASE, "labels": BEST_ROTS}    
    ##helper.visualiza_bhtsne(base_rdz, arq_log.replace(".log",""))
        
    logging.info("Tempo total do programa: %0.2f" % (round(time()-t0,2)))    
    logging.info("ENCERRAMENTO DO PROGRAMA")    
    
    

######################################################################################################
######################################################################################################
######################################################################################################

