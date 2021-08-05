# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 13:34:10 2016
@author: antoniosousa
Realiza o processo de classificacao de imagens baseado no processo
de extracao de patches das imagens. 

Classifica uma base passada

"""

from os import path
from time import time
from optparse import OptionParser
import classifica as clasf
import logging 
import helper
from datetime import datetime


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
    if clasf.existe_opt(parser, "opt_log"):
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
    if clasf.existe_opt(parser, "base_treino"):
        if not (path.isdir(options.base_treino)):
            clasf.loga_sai("Erro: Caminho da base de treino incorreto ou o arquivo nao existe.")            
    else:
        clasf.loga_sai("Base de treino ausente.")
        
    if clasf.existe_opt(parser, "base_teste"):
        if not (path.isdir(options.base_teste)):
            clasf.loga_sai("Erro: Caminho da base de teste incorreto ou o diretorio nao existe.")            
    else:
        clasf.loga_sai("Base para classificacao ausente")        
    
    # verifica se será utilizado descarte de patches
    descarta=False    
    if clasf.existe_opt(parser, "descarte"):
       descarta=True
       DESCARTA=True
       
    # Seta o valor do incremento dos patches de tamanho fixo    
    inc=ex.INCR_PATCH   
    if clasf.existe_opt(parser, "opt_incremento"):
       try: 
           inc = int(options.opt_incremento)    
       except ValueError:
           clasf.loga_sai("Valor de incremento inválido! Um inteiro era esperado.")
                          
      
    # metodo de extracao de atributos a ser utilizado
    if not(clasf.existe_opt(parser, "opt_metodo")):
       clasf.loga_sai("Erro: Metodo de extração ausente.")       
        
    # verifica o tipo de divisão de patches que sera utilizada    
    fl_altura = clasf.existe_opt(parser, "opt_altura")
    fl_nivel = clasf.existe_opt(parser, "opt_nivel")
    fl_tamanho = clasf.existe_opt(parser, "opt_tamanho")
    arqs_treino = []    # arquivos das bases de treino
    arqs_clf = []       # arquivos dos atributos das bases a classificar
    # pelo menos um tipo de divisao deve ser escolhido
    if not(fl_altura) and not(fl_nivel) and not(fl_tamanho):
       clasf.loga_sai("Erro: Pelo menos uma divisão dos patches deve ser escolhida entre tamanho, nivel da quadtree ou altura da quadtree de patches.")
    else:
        # apenas uma das opcoes deve ser passada
        if not(helper.xor_n(fl_altura,fl_nivel,fl_tamanho)):
           clasf.loga_sai("Erro: Apenas uma opção de divisão dos patches deve ser escolhida.")         
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
                clasf.loga_sai("Valor inválido, quando um inteiro era esperado. Erro: " + str(v))            
    
    if not arqs_treino:
        clasf.loga_sai("Erro ao recuperar nome dos arquivos de treino!")
    
    # loga o tempo de execução da extração
    logging.info("Tempo total de extracao: " + str(round(time()-t0,3)))
    
    # verifica as listas de arquivos            
    if len(arqs_treino) != len(arqs_clf):
       clasf.loga_sai("Divergencia entre arquivos da base de treinamento e de classificação.") 
    
    ##print("Arquivos de treino: "+str(arqs_treino))    #apenas para debugar
    ##print("Arquivos de classificação: "+str(arqs_clf))   #apenas para debugar
    ##print("Arquivos ppi: " + str(arqs_ppi_ts))    #apenas para debugar
    
    if clasf.existe_opt(parser, "opt_clf"):       
        opt_clfs = options.opt_clf.split(',')     
    else:
        ### executar apenas a extracao  GO HORSE!  ###                    
        clasf.loga_sai("Executados apenas os procedimentos de extração...") 
        
        
    # passado mais de um classificador sem um metodo de fusao definido    
    if (len(opt_clfs) > 1):
        if not(clasf.existe_opt(parser,'fusao_clf')):
            clasf.loga_sai("Erro: Metodo de fusao de classificadores ausente.")
        # verifica se os metodo de fusao definido é válido
        if not(options.fusao_clf in FUSAO):
           clasf.loga_sai("Erro: Metodo de fusao desconhecido. Valores aceitos: " + str(FUSAO)) 
        
    # verifica se os classificadores passados são validos
    for c in opt_clfs:
        if not(CLFS[c]):
           clasf.loga_sai("Erro: Classificador desconhecido. Valores aceitos: " + str(CLFS.keys))                
    
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