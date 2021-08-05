# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 23:21:04 2016

@author: willian

Adaptado dos códigos disponíveis em https://github.com/DEAP/notebooks

Execução do programa:

- Recebe uma base de atritubos (PFTAS) já extraidos, de três (03) diretórios diferentes:
     treino - validacao - teste
- Utiliza um algoritmo genetico para selecionar as instâncias que conduzem a 
  melhor taxa de classificação com maior descarte de patches possível, utilizando uma 
  base de validacao para isso
- A partir do melhor individuo obtido do algoritmo genético, gera uma nova base para
  utilizar na classificação 
- Treina um classificador com essa base e aplica sobre a base de testes e coleta os resultados
"""

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import classifica 
import logging
from os import path
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score,roc_curve, auc
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import StratifiedKFold
from sklearn import tree
from scipy import interp
from deap import creator, base, tools, algorithms
from time import time
from datetime import datetime
from optparse import OptionParser

'''
Carrega uma base a partir de um arquivo .svm
'''
def carrega_base(arq_base):
    atribs = None 
    rotulos = None             
    #TODO: a quantidade de atributos está sendo passada diretamente para a funcao de carga
    # do arquivo, mas deveria funcionar de modo automatico
    # Da forma como esta codificado, funciona apenas para PFTAS aplicados em canais RGB,
    # concatenado com o vetor de atributos (dimensao 81) negado. 
    atribs, rotulos = load_svmlight_file(arq_base, dtype=np.float32,n_features=162) 
    
    return (atribs, rotulos)

'''
Carrega a linha de atributos da base de testes indicadas pelo valor - 1 ou 0 -
no agrupamento. 1 = relevante, 0 = irrelevante
'''
def carrega_agrup(agrupamento, base_tr, rotulos_tr):    
    mascara = []    
    for b in agrupamento:
        if b == 1:
            mascara.append(True)
        else:
            mascara.append(False)
    
    mascara = np.asarray(mascara)    
    atribs_uso = base_tr[mascara, :]
    
    m = np.ma.array(rotulos_tr, mask=~mascara)    
    rotulos_uso = m.compressed()    
    
    return (atribs_uso, rotulos_uso)

'''
Avalia a taxa de classificacao para os patches (linhas de atributos) selecionados
através do agrupamento.
Os dois valores de fitness devolvidos são (o valor da taxa de reconhecimento, total de 1's do agrupamento)
'''
def avalia_agrupamento(agrupamento):    
    
    base_tr = BASE_TR
    rotulos_tr = ROTULOS_TR    
    base_val = BASE_VAL
    rotulos_val = ROTULOS_VAL
    
    # Caso o agrupamento não utilize nenhum patch
    if (sum(agrupamento) == 0):
        #return (0.0, 0.99)
        return (0.0, )
    
    # Carrega apenas os patches que sao relevantes para fazer treinamento
    # do classificador        
    base_uso, rotulos_uso = carrega_agrup(agrupamento, base_tr, rotulos_tr)    
    
    # Verifica as classes presentes no agrupamento carregado
    # Classificadores como SVM necessitam de pelo menos duas classes presentes
    # na base utilizada para treinamento do classificador
    if sum(rotulos_uso) == 0:  # apenas instancias da classe 0 estao presentes
       return (0.0, )
    elif sum(rotulos_uso) == len(rotulos_uso): # apenas instancias da classe 1 estao presentes
       return (0.0, )  
    else:    
        # Treina um classificador usando apenas os exemplares definidos pelo 
        # agrupamento
        clf = classifica.get_clf("svm")             
        clf.fit(base_uso, rotulos_uso)
        r = clf.predict(base_val)
        
            
        # Plota resultados da curva ROC obtida
        #plot_roc(rotulos_val, r)
        return (roc_auc_score(rotulos_val, r),)
        # Para avaliação multi-objetivo utilizar a linha abaixo
        #return (accuracy_score(rotulos_val, r), 1-sum(agrupamento)/MAX_AGRUPA)

def plot_roc(y, probas):
    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    
    color = 'cyan'
    lw = 2
        
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y, probas)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    #roc_auc = auc(fpr, tpr)
    roc_auc = roc_auc_score(y, probas)
    plt.plot(fpr, tpr, lw=lw, color=color, label='ROC (area = %0.2f)' % (roc_auc))
            
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Luck')
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    
def grava_melhor_base(agrupamento):
    base_tr = BASE_TR
    rotulos_tr = ROTULOS_TR
    arq_treino = ""
    
    if (sum(agrupamento) == 0):
       logging.info("Erro: Agrupamento de tamanho 0 - nenhum patch selecionado")
    
    # Carrega apenas os patches que sao relevantes para fazer treinamento
    # do classificador        
    atributos, rotulos = carrega_agrup(agrupamento, base_tr, rotulos_tr)    
    
    if len(atributos) > 0:
       dump_svmlight_file(atributos, rotulos, arq_treino)   
    else:
       logging.info("Erro ao processar a base. Não há linhas!") 

def make_bool(x):
    return (x)
    
def exibe_results_multiobjetivo(pop, log, hof):  
         
    for idx,best in enumerate(hof):
        #print("Quantidade de 1's do melhor %d individuo: %d" % (idx, sum(best)))
        logging.info("Percentual de redução do %dº melhor individuo: %.2f" % (idx, 1-sum(best)/len(best)))
        logging.info("%dº melhor individuo com fitness: %.2f" % (idx, best.fitness.values[0]))
    
    gen, avg, min_, max_ = log.select("gen", "avg", "min", "max")
    
    avg_rec = []
    min_rec = []
    max_rec = []
    avg_desc = []
    min_desc = []
    max_desc = []
    
    for a,b,c in zip(avg,min_,max_):
       avg_rec.append(a[0])
       avg_desc.append(a[1])
       min_rec.append(b[0])
       min_desc.append(b[1])
       max_rec.append(c[0])
       max_desc.append(c[1])
    
    f, axarr = plt.subplots(2, sharex=True)    
    
    # primeiro grafico - tx de reconhecimento
    axarr[0].plot(gen, avg_rec, label="média")
    axarr[0].plot(gen, min_rec, label="mínimo")
    axarr[0].plot(gen, max_rec, label="máximo")
    plt.xlabel("Geração")    
    axarr[0].legend(loc="lower right")
    axarr[0].set_title('Taxa Reconhecimento')    
    
    # segundo grafico - tx de descarte
    axarr[1].plot(gen, avg_desc, label="média")
    axarr[1].plot(gen, min_desc, label="mínimo")
    axarr[1].plot(gen, max_desc, label="máximo")    
    axarr[1].legend(loc="lower right")
    axarr[1].set_title('Taxa Descarte')    
    
    plt.savefig("grafico.pdf")
    plt.show()    

def exibe_results(pop, log, hof):  
         
    for idx,best in enumerate(hof):
        #print("Quantidade de 1's do melhor %d individuo: %d" % (idx, sum(best)))
        logging.info("Percentual de redução do %dº melhor individuo: %.2f" % (idx+1, 1-sum(best)/len(best)))
        logging.info("%dº melhor individuo com fitness: %.2f" % (idx+1, best.fitness.values[0]))
    
    gen, avg, min_, max_ = log.select("gen", "avg", "min", "max")
    
    avg_auc = []
    min_auc = []
    max_auc = []    
    
    for a,b,c in zip(avg,min_,max_):
       avg_auc.append(a[0])       
       min_auc.append(b[0])       
       max_auc.append(c[0])       
    
    # primeiro grafico - tx de reconhecimento
    plt.plot(gen, avg_auc, label="média")
    plt.plot(gen, min_auc, label="mínimo")
    plt.plot(gen, max_auc, label="máximo")
    plt.xlabel("Geração")    
    plt.legend(loc="lower right")
    #plt.set_title('AUC')    
    
    plt.savefig("grafico.pdf")
    plt.show()

    
# Executa um algoritmo genetico sobre a populacao passada                   
def exec_ssga(pop, toolbox, tx_cross, tx_mutacao, geracoes, stats, hof):
    
    # avalia a população inicial    
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    hof.update(pop)
    r = stats.compile(pop)                        
    r["gen"] = 0

    logbook = tools.Logbook()    
    logbook.record(**r)
    
    for g in range(geracoes):        
        
        # Seleciona dois pais da geracao atual
        pais = toolbox.select(pop, 2)
        # Clone the selected individuals
        filhos = [toolbox.clone(ind) for ind in pais] 

        # Realiza crossover 
        if random.random() < tx_cross:
            toolbox.mate(filhos[0], filhos[1])
            del filhos[0].fitness.values
            del filhos[1].fitness.values
        
        # Aplica mutacao
        for mutant in filhos:
            if random.random() < tx_mutacao:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Avalia o fitness dos filhos gerados
        #invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, filhos)
        for ind, fit in zip(filhos, fitnesses):
            ind.fitness.values = fit

        # Entre os filhos gerados e os pais, apenas os melhores são mantidos
        idx1 = pop.index(pais[0])        
        idx2 = pop.index(pais[1])
        
        #l = sorted(pais+filhos, key=lambda indv: indv.fitness.values)   
        melhores = [(best, 1-sum(best)/len(best)) for best in pais+filhos]
        l = sorted(melhores, key=lambda k:(k[1],k[0].fitness.values[0]))
        pop[idx1] = l[0][0]
        pop[idx2] = l[1][0]
        
        hof.update(pop)
        r = stats.compile(pop) 
        r["gen"] = g+1                       
        logbook.record(**r)
    
    return (pop,logbook,hof)    


# Cria população inicial 
def populacao_inicial(toolbox, tamanho=10):
    pop = toolbox.population(n=1)    
    #pop.append(toolbox.indiv_1())
    tudo_zero = toolbox.indiv_0()        
    tudo_zero[random.randint(0,MAX_AGRUPA)] = 1 # acrescenta 1 patch ao agrupamento vazio
    pop[0] = tudo_zero
    
    for i in range(int(tamanho/5)):        
        tudo_zero = toolbox.indiv_0()        
        tudo_zero[random.randint(0,MAX_AGRUPA)] = 1 # acrescenta 1 patch ao agrupamento vazio
        pop.append(tudo_zero)         
        
        t1 = toolbox.indiv_0()
        t2 = toolbox.indiv_1()
            
        t = toolbox.mate(t1, t2)    
        if (sum(t[0]) > 0):
            pop.append(t[0])
            
        if (sum(t[1]) > 0):    
            pop.append(t[1])    
    
        t1 = toolbox.mutate(toolbox.indiv_0(), indpb=i/100)[0]        
        t2 = toolbox.mutate(toolbox.indiv_1(), indpb=i/100)[0]
        
        if (sum(t1) > 0):
            pop.append(t1)
            
        if (sum(t2) > 0):    
            pop.append(t2)
    
    return (pop)

'''
Cria um indivíduo com apenas uma posição válida escolhida aleatoriamente
'''
def cria_indiv_1pos():
    # adiciona um individou que tem apenas uma posicao valida
    tudo_zero = toolbox.indiv_0()        
    tudo_zero[random.randint(0,MAX_AGRUPA)] = 1 # acrescenta aleatoriamente 1 patch ao agrupamento vazio
    return (tudo_zero)    
    
'''
Verifica se uma opção passada existe na lista de argumentos do parser
'''  
def existe_opt (parser, dest):
   if any (opt.dest == dest and (opt._long_opts[0] in sys.argv[1:] or opt._short_opts[0] in sys.argv[1:]) for opt in parser._get_all_options()):
      return True
   return False 

# Insere a informação do erro no log do programa e forca saida, encerrando o programa
def loga_sai(erro):
    logging.info(erro)
    sys.exit(erro)
    
if __name__ == "__main__":
    t0 = time()
    
    ### CARREGA OPCOES PASSADAS     
    parser = OptionParser()
    parser.add_option("-T", "--base-treino", dest="base_treino",
                      help="Localização do arquivo da base de treino")
    parser.add_option("-t", "--base-teste", dest="base_teste",
                      help="Localização do arquivo da base de teste") 
    parser.add_option("-v", action="store_true", dest="verbose", help="Exibir saida no console.") 
                      
    (options, args) = parser.parse_args()   
    
    
    ## Cria a entrada de log do programa
    idarq=datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')  
    
    logging.basicConfig(filename='genpatch-'+idarq+'.log', format='%(message)s', level=logging.INFO)    
    
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
    #id_arq = arq_base_tr.split("/")[-4]            
    logging.info("Base de Treino: " + arq_base_tr)    
    BASE_TR_INI, ROTULOS_TR_INI = carrega_base(arq_base_tr)
    
    # carrega base de testes
    arq_base_ts = options.base_teste    
    #logging.info("Base de Teste: " + arq_base_ts)    
    #BASE_TS, ROTULOS_TS = carrega_base(arq_base_ts)

    # Faz a divisão da base de treino em folds
    skf = StratifiedKFold(n_splits=3)    
    
    melhores = []   # armazena informação das execuções a serem realizadas
    for tr_idx, ts_idx in skf.split(BASE_TR_INI, ROTULOS_TR_INI):        
        BASE_TR, BASE_VAL = BASE_TR_INI[tr_idx], BASE_TR_INI[ts_idx]
        ROTULOS_TR, ROTULOS_VAL = ROTULOS_TR_INI[tr_idx], ROTULOS_TR_INI[ts_idx]
    
        #MAX_AGRUPA = BASE_TR.shape[0] 
        MAX_AGRUPA = len(tr_idx) 
    
        # cria população inicial        
        # A PRIMEIRA PROPOSTA USA como fitness a maximizacao da classificacao e maximizacao do descarte (minimizar a quantidade de patches relevantes)
        # creator.create("FitAccDesc", base.Fitness, weights=(1.0,1.0))
        # A SEGUNDA PROPOSTA APÓS CONVERSA COM O ORIENTADOR USA a area sob a curva (AUC) de uma curva ROC como fitness
        creator.create("FitAUC", base.Fitness, weights=(1.0,))
        creator.create("Agrupamento", list, fitness=creator.FitAUC)
        
        toolbox = base.Toolbox()    
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("bit_1", make_bool, 1)
        toolbox.register("bit_0", make_bool, 0)    
        
        toolbox.register("indiv_1", tools.initRepeat, creator.Agrupamento, toolbox.bit_1, MAX_AGRUPA)    
        toolbox.register("indiv_0", tools.initRepeat, creator.Agrupamento, toolbox.bit_0, MAX_AGRUPA)    
            
        toolbox.register("individual", tools.initRepeat, creator.Agrupamento, toolbox.attr_bool, MAX_AGRUPA)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        toolbox.register("evaluate", avalia_agrupamento)    
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.3)    
        toolbox.register("select", tools.selTournament, tournsize=5)    
        
        # PROGRAMA PRINCIPAL    
        GERACOES = 25
        TX_MUTACAO = 0.05
        TX_CROSS = 0.5
        logging.info("Tamanho dos agrupamentos: %d" %(MAX_AGRUPA))       
        
        # Cria e inicia a população
        pop = populacao_inicial(toolbox, tamanho=20)
            
        for idx,indv in enumerate(pop):
            logging.info("Quantidade de 1's do individuo %d: %d" % (idx+1, sum(indv)))
            
        hof = tools.HallOfFame(2)    
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)    
        stats.register("avg", np.mean, axis=0) 
        stats.register("min", np.min, axis=0) 
        stats.register("max", np.max, axis=0)    
        #pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=TX_CROSS, mutpb=TX_MUTACAO, ngen=GERACOES, stats=stats, halloffame=hof, verbose=True)
        t1 = time()
        pop, logbook, hof = exec_ssga(pop, toolbox, TX_CROSS, TX_MUTACAO, GERACOES, stats, hof)        
        logging.info("Tempo de execução para 1 fold: %0.2f" % (round(time()-t1,2)))
        #exibe_results(pop, logbook, hof)
        #melhores.append((hof, tr_idx))    
        melhores += [(best, tr_idx, 1-sum(best)/len(best)) for best in hof]
    logging.info("Tempo total de seleção do AG: %0.2f" % (round(time()-t0,2)))    
    
    #for m in melhores:
    #    print(str(sum(m[0]))+" - "+str(m[0].fitness.values[0])+" - "+str(m[2]))
    #print("--------------------------------")    
    melhores = sorted(melhores, key=lambda k:(k[2],k[0].fitness.values[0]), reverse=True)
    
    #for m in melhores:
    #    print(str(sum(m[0]))+" - "+str(m[0].fitness.values[0])+" - "+str(m[2]))
    
    # Executa classificação de teste
    # Carrega os agrupamentos com melhor desempenho e utiliza ele para classificar
    # uma base de testes, com imagens novas ao processo de classificação
    resultados = []
    for (best,idx,reducao) in melhores:        
        t2 = time()
        base_best, rotulos_best = carrega_agrup(best, BASE_TR_INI[idx], ROTULOS_TR_INI[idx])    
    
        # Treina um classificador usando apenas os exemplares definidos pelo 
        # agrupamento
        #clf = classifica.get_clf("dt")     	
        r = classifica.classificacao_base(base_best, rotulos_best, arq_base_ts, arq_base_ts[:arq_base_ts.find(".svm")]+".ppi", "svm")
        resultados.append(r)
        '''
        clf.fit(base_best, rotulos_best)
        r_pred = clf.predict(BASE_TS)
        
        # cria as matrizes de confusao
        cm = confusion_matrix(ROTULOS_TS, r_pred)
        logging.info("\nMatriz de confusao: ") 
        logging.info(cm) 
        
        # exibe a taxa de classificacao
        r_pred = np.asarray(r_pred) 
        r_tst = np.asarray(ROTULOS_TS) 
        taxa_clf = np.mean(r_pred.ravel() == r_tst.ravel()) * 100 
        '''
        classifica.processa_resultados(resultados, ["."], ["AG","svm"], idarq)
        logging.info("Taxa de Classificação: %f " % (round(taxa_clf,3)))    
        logging.info("Tempo de classificação: %0.2f" % (round(time()-t2,2)))    
        
    logging.info("Tempo total do programa: %0.2f" % (round(time()-t0,2)))    
    logging.info("ENCERRAMENTO DO PROGRAMA")    
    #print(str(melhores))
