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
  a base final obtida. ll
  
"""

import numpy as np
import classifica
import logging
import helper
import extrator as ex
import matplotlib.pyplot as plt
import multiprocessing as mp

from os import path
from scipy import interp
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from deap import creator, base, tools
from time import time
from datetime import datetime
from optparse import OptionParser
from helper import loga_sai, existe_opt
import linecache
import sys
from os import makedirs
import cv2
from matplotlib.patches import Rectangle

# Qtd de folds a considerar ao avaliar a base filtrada por uma particula
# do PSO
K_FOLDS=3

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

LEGENDAS = ["Energia",
            "Contraste",
            "Correlacao",
            "Variancia",
            "Homogeneidade",
            "Media",
            "Variancia",
            "Entropia Total",
            "Entropia",
            "Variancia da Diferenca",
            "Entropia da Diferenca",
            "IMC1",
            "IMC2",
            "CCM"]

ATR_GLCM = {
                0: {'atr' : "Energia",       'valores' : [],  'tam_bases' : []}, 
                1: {'atr' : "Contraste",     'valores' : [],  'tam_bases' : []}, 
                2: {'atr' : "Correlacao",    'valores' : [],  'tam_bases' : []}, 
                4: {'atr' : "Homogeneidade", 'valores' : [], 'tam_bases' : []}, 
                5: {'atr' : "Media",         'valores' : [],  'tam_bases' : []}, 
                8: {'atr' : "Entropia",      'valores' : [],  'tam_bases' : []}
            }

IDX_FILTRO = [0]
M_BASE = []




# Lista de posicoes do espaco de busca que ja foram visitadas por alguma
# particula do PSO. Essa lista sera usada para evitar que o mesmo local do
# espaco de busca seja visitado mais de uma vez por diferentes particulas
VISITADOS = []

# Para buscar dentro da fronteira o melhor valor entre AUC e tamanho da base de treino use a linha abaixo
##creator.create("FitnessAUC", base.Fitness, weights=(1.0,-1.0))
# Para realizar a busca apenas para AUC, utilize a linha abaixo. Se a linha abaixo fornecer valores melhores
# que a base de treinamento completa, porém gerando uma base menor, terá ocorrido a eliminação de patches
# prejudiciais
creator.create("FitnessAUC", base.Fitness, weights=(1.0,))

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
def filtra_base(base_tr, rotulos_tr, atrs_glcm, idx_filtro, val_filtro):    
    # Verificações de tamanho das matrizes    
    if base_tr.shape[0] == 0 or rotulos_tr.shape[0] == 0:
        loga_sai("A base a ser filtrada não pode ser vazia")
    elif not (base_tr.shape[0] == rotulos_tr.shape[0]):
        loga_sai("Quantidade de instâncias e rótulos diferem")
    
    # Filtra a base passada            
    idxs = []
    linhas = atrs_glcm.shape[0]
    
    for l in range(linhas):         
        condicao = atrs_glcm[l][idx_filtro] < val_filtro
        idxs.append(condicao)        
                
    idxs = np.where(idxs)[0]
    base_filt = base_tr[idxs, :]
    rotulos_filt = rotulos_tr[idxs]    
    
    return (base_filt, rotulos_filt, idxs)

def tam_base_filtro(base_tr, rotulos_tr, atrs_glcm, idx_filtro, val_filtro):    
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
        condicao = atrs_glcm[l][idx_filtro] < val_filtro
        idxs.append(condicao)        
                
    idxs = np.where(idxs)[0]
    base_filt = base_tr[idxs, :]
    rotulos_filt = rotulos_tr[idxs]
        
    return (len(rotulos_filt))

    
'''
Avalia a taxa de classificacao obtida a partir da base de treinamento filtrada, 
utilizando os parâmetros da particula para realizar a filtragem da base.
Os valores da particulas representam valores dos atributos de Haralick extraídos
anteriormente para cada patch da base.
Os valores utilizados para filtragem serão (em ordem): ["Entropy", "Correlation"]                   
'''
def avalia_particula(particula):
       
    base_tr = BASE_TR
    rotulos_tr = ROTULOS_TR    
    base_glcm = BASE_GLCM
    
    # Atualiza lista de visitados
    VISITADOS.append(particula[0])
    
    # Filtra a base baseado nos valores da particula
    base_uso, rotulos_uso, idx_uso = filtra_base(base_tr, rotulos_tr, base_glcm, filtro=particula)    
    
    if rotulos_uso.shape[0] < 0.15*rotulos_tr.shape[0]:
       logging.info("Base filtrada muito pequena: {0} linhas.".format(rotulos_uso.shape[0]))  
       return (0.0, )
        
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
    
    return (round(mean_auc,4), )   
    
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
    #clf = svm.SVC(gamma=0.5, C=32, cache_size=500, probability=True)  
    clf = classifica.get_clf(id_clf)
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


def filtra_clasf(BASE_TR, ROTULOS_TR, BASE_GLCM, idx_glcm, val_filtro):
    base_filt, rots_filt, idxs_filt = filtra_base(BASE_TR, ROTULOS_TR, BASE_GLCM, idx_glcm, val_filtro)
    tam_base = rots_filt.shape[0]
    #tam_bases.append(tam_base/1000)
    
    # Valores a serem retornados
    taxa_base = 0.
    auc_base = 0.
    
    # Treina um classificador usando apenas os exemplares definidos pela pela melhor particula    
    if tam_base > 0:
        r = classificacao_probas(base_filt, rots_filt, base_teste, base_teste.replace(".svm",".ppi"), "rf")    
        
        if not r is None:
            logging.info("Atributo: {0} Valor: {1} Tam. Base: {2} AUC: {3} Tx. Clf.: {4}".format(dados_glcm['atr'], val_filtro, tam_base, r['auc'], r['taxa_clf']))     
            taxa_base = r['taxa_clf']
            auc_base = r['auc']
        else:
            logging.info("Atributo: {0} Valor: {1} Tam. Base: {2} AUC: {3} Tx. Clf.: {4}".format(dados_glcm['atr'], val_filtro, tam_base, 0, 0))                             
    else:
        logging.info("Atributo: {0} Valor: {1} Tam. Base: {2} AUC: {3} Tx. Clf.: {4}".format(dados_glcm['atr'], val_filtro, tam_base, 0, 0))     
        
    return (taxa_base, auc_base, tam_base/1000)


def marca_patches_n(arq_base, qid_patches, legendas, tam_patch=64, rgb=False, path_saida="plots/"):    
    
    if type(qid_patches) is int:
        qid_patches = [qid_patches]
    elif not type(qid_patches) is list:
        sys.exit("Esperado lista de inteiros ou inteiro.")
    
    arq_ppi = arq_base.replace(".svm",".ppi")
    print(arq_ppi)
    if path.isfile(arq_ppi):
        #linha_ppi = str(linecache.getline(arq_ppi, 1)).split(',')   # primeira linha
        linha_ppi = linecache.getline(arq_ppi, 1).replace('"','').replace('\n','').split(',')   # primeira linha
    else:
        sys.exit("Arquivo Inexistente: {0}".format(arq_ppi))

    imgs_patches = {}
    for qid_patch, legenda in zip(qid_patches, legendas):
        ppi = int(linha_ppi[1].replace('"',''))    
        n_img = qid_patch//ppi    
        print("Numero imagem: {0}".format(n_img))
        
        linha_ppi = linecache.getline(arq_ppi, n_img+1).replace('"','').replace('\n','').split(',')   # linha da imagem passada
        print("Linha PPI: {0}".format(linha_ppi))
        arq_img = linha_ppi[-1]
            
        n_patch = qid_patch%ppi
        print("Numero Patch: {0}".format(n_patch))
        
        if not n_img in imgs_patches.keys():
            imgs_patches[n_img] = {'arquivo': arq_img, 'patches': [n_patch], 'legendas': [legenda] }
        else:
            imgs_patches[n_img]['patches'].append(n_patch) 
            imgs_patches[n_img]['legendas'].append(legenda) 

    for n_img, valores in imgs_patches.items(): 
        arq_img = valores['arquivo']
        n_patches = valores['patches']
        legendas = valores['legendas']
        
        if path.isfile(arq_img):
            if rgb:
                img = cv2.imread(arq_img)  # imagem original (rgb)
            else:
                img = cv2.imread(arq_img, cv2.IMREAD_GRAYSCALE)  # imagem original escala de cinza
        else:
            sys.exit("Arquivo de imagem incorreto - {0}".format(arq_img))
            
        patches = ex.cria_patches(img, tam_patch, tam_patch, rgb)
            
        tam_x = int(img.shape[0]/tam_patch)
        tam_y = int(img.shape[1]/tam_patch)
        
        # Get current size
        fig_size = plt.rcParams["figure.figsize"]

        # Set figure width to 12 and height to 9
        fig_size[0] = 12
        fig_size[1] = 9
        plt.rcParams["figure.figsize"] = fig_size
        
        fig,axes = plt.subplots(tam_x,tam_y)
            
        ax = axes.ravel()        
        i = 0
        nx = 0
        ny = 0
        for i in range(patches.shape[0]):  
            patch = patches[i]
            ax[i].set_xticklabels([])
            ax[i].set_yticklabels([]) 
            ax[i].axis('off')
    
            #if entr_media < 5.5:
            #    patch[patch != 0] = 0            
            
            if (rgb):
                ax[i].imshow(patch)
            else:
                ax[i].imshow(patch, cmap="gray")
            
            v = (ax[i].get_xlim()[1]-ax[i].get_xlim()[0])
            nx = int((i%tam_x)*tam_patch) + int(tam_patch/2)        
            nx = int(nx/v + v/2)  
            ny = int(i/tam_x)*tam_patch + int(tam_patch/2)
            ny = int(ny/v + v/2)  
            
            if i in n_patches:    
                ax[i].text(nx,ny, str(legenda),color='r',ha='center',va='center',fontsize='18')                 
                axAxis = ax[i].axis()
                bordaX0 = axAxis[0]-0.7
                bordaY0 = axAxis[2]-0.2
                bordaX1 = (axAxis[1]-axAxis[0])+1     
                bordaY1 = (axAxis[3]-axAxis[2])+0.4                          
                rec = Rectangle((bordaX0,bordaY0),bordaX1,bordaY1,fill=False,lw=3, edgecolor='darkgreen')
                rec = ax[i].add_patch(rec)
                rec.set_clip_on(False)
            '''    
            else:
                #ax[i].text(nx,ny,str(i),color='black',ha='center',va='center',fontsize='12') 
                
                axAxis = ax[i].axis()
                bordaX0 = axAxis[0]-0.7
                bordaY0 = axAxis[2]-0.2
                bordaX1 = (axAxis[1]-axAxis[0])+1     
                bordaY1 = (axAxis[3]-axAxis[2])+0.4                          
                rec = Rectangle((bordaX0,bordaY0),bordaX1,bordaY1,fill=False,lw=1, edgecolor='r')
                rec = ax[i].add_patch(rec)
                rec.set_clip_on(False)
            '''
        fig.subplots_adjust(wspace=0.05, hspace=0.05)        
        dir_imagem = path.dirname(arq_img)
        nome_imagem = path.basename(arq_img)
        dir_saida = path_saida + "/".join(dir_imagem.split('/')[-3:])+"/"
        if not path.isdir(dir_saida):
            makedirs(dir_saida)
            
        img_saida = dir_saida + nome_imagem
        plt.savefig(img_saida)
        plt.close('all')
        #plt.show()
        
        #return (entrops_patches)    
        


# PROGRAMA PRINCIPAL    
###################################################################################################################

if __name__ == "__main__":
    toolbox = base.Toolbox()
    toolbox.register("particle", generate, size=1, pmin=0.01, pmax=0.999, smin=-2, smax=2)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", updateParticle, phi1=0.2, phi2=0.8)
    toolbox.register("evaluate", avalia_particula)

    t0 = time()    
    ### CARREGA OPCOES PASSADAS     
    
    parser = OptionParser()    
    parser.add_option("-v", action="store_true", dest="verbose", help="Exibir saida no console.")
    parser.add_option("-l", "--log", dest="opt_log", help="Arquivo de log a ser criado.")
    
                      
    (options, args) = parser.parse_args()   
    
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
    
    #bases = helper.todas_breakhis()
    bases = helper.todas(40)
    # Apenas para testes
    
    # Gera as legendas    
    leg = [LEGENDAS[i] for i in IDXS_GLCM]
       
    
    for mag, folds in bases.items():        
        for fold, bases in folds.items():
            vals_glcm = []
            tams_glcm = []
            taxas_fold = []
            aucs_fold = []
            base_treino = bases['tr']['pftas']
            base_teste = bases['ts']['pftas']
            
            # verifica se a base de treino e de teste passadas existem    
            if not (path.isfile(base_treino)):
                loga_sai("Erro: Caminho da base de treino incorreto ou o arquivo nao existe.")
                    
            
            if not (path.isfile(base_teste)):
                loga_sai("Erro: Caminho da base de teste incorreto ou o diretorio nao existe.")  
                
            ### CONFIGURACOES INICIAIS E CARGA DAS BASES    
            # Carrega base de treino
            arq_base_tr = base_treino     
            arq_base_glcm = arq_base_tr.replace(".svm",".glcm")           
                        
            BASE_GLCM, _,QID_GLCM = carrega_base(arq_base_glcm,n_features=14)   
            
            BASE_GLCM = BASE_GLCM.toarray()
            BASE_GLCM = MinMaxScaler().fit_transform(BASE_GLCM)
            
            minimos = BASE_GLCM.min(axis=0)
            maximos = BASE_GLCM.max(axis=0)            
            
            for idx in IDXS_GLCM:    
                linhas_patches = np.where(BASE_GLCM[0:50,idx] < 0.6)
                ATR_GLCM[idx]['patches'] = QID_GLCM[linhas_patches].tolist()
                marca_patches_n(arq_base_tr, ATR_GLCM[idx]['patches'], tam_patch=64, rgb=True, path_saida="plots/") 
            
            
            
            '''    
            # para os intervalos gerados para cada um dos atributos,
            # filtra a base de treino e avalia os tamanhos das bases geradas
            for idx_glcm, dados_glcm in ATR_GLCM.items():                
                clf_auc = []
                with mp.pool.Pool(5) as p:
                    for val_filtro in dados_glcm['valores']:
                        
                        proc = p.apply_async(filtra_clasf, (BASE_TR, ROTULOS_TR, BASE_GLCM, idx_glcm, val_filtro))
                        clf_auc.append(proc.get())

                taxas_bases, aucs_bases, tams_bases = zip(*clf_auc)                                
                taxas_fold.append(list(taxas_bases))
                aucs_fold.append(list(aucs_bases))
                tams_glcm.append(list(tams_bases))
                vals_glcm.append(ATR_GLCM[idx_glcm]['valores'])
            
            
            helper.plota_grafico(vals_glcm, tams_glcm, arquivo="{0}-{1}-tam".format(mag,fold), titulo="{0} - {1} Qtd. Patches".format(mag,fold), tituloX="Atrib. GLCM", tituloY="Tam. da Base (x1000)", legendas=leg)
            helper.plota_grafico(vals_glcm, taxas_fold, arquivo="{0}-{1}-clf".format(mag,fold), titulo="{0} - {1} Classificação".format(mag,fold), tituloX="Atrib. GLCM", tituloY="Tx. Classificação", legendas=leg)
            helper.plota_grafico(vals_glcm, aucs_fold, arquivo="{0}-{1}-auc".format(mag,fold), titulo="{0} - {1} AUC".format(mag,fold), tituloX="Atrib. GLCM", tituloY="AUC", legendas=leg)
            '''
  
    logging.info("Tempo total do programa: %0.2f" % (round(time()-t0,2)))    
    logging.info("ENCERRAMENTO DO PROGRAMA")    
    
    

######################################################################################################
######################################################################################################
######################################################################################################

