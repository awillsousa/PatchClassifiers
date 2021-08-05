# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 13:34:10 2016
@author: antoniosousa
Procedimentos de extracao de caracteristicas de imagens, utilizando como refereência
métodos de extração de patches.

O programa realiza a classificacao de uma base de imagens baseando-se nos atributos extraidos
dos patches das imagens

"""
import sys
import logging
import datetime
import ConfVars
from Classificador import Classificador
from os import path
from time import time
from optparse import OptionParser

# PROGRAMA PRINCIPAL 
def main():
    t0 = time()
    
    uso = "Uso: %prog [options]"
    parser = OptionParser(uso)
    parser.add_option("-T", "--base-treino", dest="base_treino",
                      help="Localização do arquivo da base de treino")
    parser.add_option("-t", "--base-teste", dest="base_teste",
                      help="Localização do arquivo da base de teste")     
    parser.add_option("-C", "--clf", dest="clf",
                      type='choice', action='store',
                      choices=ConfVars.CLFS,
                      default=ConfVars.CLF,
                      help="Lista de classificadores a serem utilizados. [dt, qda, svm, knn]")
    parser.add_option("-d", "--destino", dest="dir_dest", default="logs/",
                      help="Diretório de gravação dos arquivos gerados durante a classificacao")  
    parser.add_option("-p", "--prefixo", dest="prefixo",
                      default='ptx',
                      help="Prefixo dos arquivos a serem gerados")
    parser.add_option("-l", "--log", dest="log", help="Arquivo de log a ser criado.")
    parser.add_option("-v", action="store_true", dest="verbose", help="Exibir saida no console.")
     
    (options, args) = parser.parse_args()
        
    # Dicionario de configuracoes do processo de extracao
    confs = {}
    
    ## Cria a entrada de log do programa
    if not(options.log):
        idarq = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')      
    else:
        idarq = options.log
        
    arq_log = "logs/{0}-{1}.log".format(options.prefixo, idarq)
    
    try:
        logging.basicConfig(filename=arq_log, format='%(message)s', level=logging.INFO)
    except FileNotFoundError as e:
        sys.exit("Arquivo de log ou caminho do arquivo incorreto.")
        
    confs['log'] = arq_log
    
    if options.verbose:
        # Configura para o log ser exibido na tela
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)            
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)   
        logging.getLogger('').addHandler(console)    
    logging.info("INICIO DO PROGRAMA") 
    logging.info("Processando parametros passados...")
    
    # verifica se o diretorio de destino existe        
    if options.dir_dest and (not (path.isdir(options.dir_dest))):
       parser.error("Caminho do diretorio de destino incorreto ou inexistente.")
    
    confs['dir_destino'] = options.dir_dest
    
    # verifica se a base de treino passada existe
    if not(options.base_treino) or not(path.isfile(options.base_treino)):
       parser.error("Caminho da base de treino passada incorreto ou inexistente.")
    
    confs['base_treino'] = options.base_treino
         
    # verifica se a base a classificar existe
    if not(options.base_teste) or not(path.isfile(options.base_teste)):
       parser.error("Caminho da base de teste passada incorreto ou inexistente.")   
    
    confs['base_teste'] = options.base_teste  
   
    confs['clf'] = options.clf
         
    logging.info("<<<<<<<< EXECUTA EXTRAÇÃO >>>>>>>>")
    t1 = time()
    
    for chave, item in confs.items():
        logging.info("{0} : {1}".format(str(chave), str(item)))
    
    patchClf = Classificador(confs)
    patchClf.executa()
            
    logging.info("\nTempo total da classificacao: " + str(round(time()-t1,3)))
    
    # Encerramento e contagem de tempo de execucao 
    logging.info("Tempo total do programa: " + str(round(time()-t0,3)))
    logging.info("ENCERRAMENTO DO PROGRAMA \n\n")         
            
    
# Chamada programa principal  
if __name__ == "__main__":	
	main()