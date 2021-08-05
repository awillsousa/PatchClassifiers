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
    
    clf = "rf"        
    dir_dest = "/logs"
    
    execs = [{'tr': "", 'ts': "", 'prefixo': ""}]
    classfs = []
    for d in execs:
    
        base_treino = d['tr']
        base_teste = d['ts']
        prefixo = d['prefixo']
        
        # Dicionario de configuracoes do processo de extracao
        confs = {}
        
        ## Cria a entrada de log do programa    
        idarq = "bkh04-clf"
            
        arq_log = "logs/{0}-{1}.log".format(prefixo, idarq)
        
        try:
            logging.basicConfig(filename=arq_log, format='%(message)s', level=logging.INFO)
        except FileNotFoundError as e:
            sys.exit("Arquivo de log ou caminho do arquivo incorreto.")
            
        confs['log'] = arq_log
        
        
        # Configura para o log ser exibido na tela
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)            
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)   
        logging.getLogger('').addHandler(console)    
        logging.info("INICIO DO PROGRAMA") 
        logging.info("Processando parametros passados...")
        
        # verifica se o diretorio de destino existe            
        confs['dir_destino'] = dir_dest
        
        # verifica se a base de treino passada existe    
        confs['base_treino'] = base_treino
             
        # verifica se a base a classificar existe
        confs['base_teste'] = base_teste  
       
        confs['clf'] = clf
             
        logging.info("<<<<<<<< EXECUTA EXTRAÇÃO >>>>>>>>")    
        
        for chave, item in confs.items():
            logging.info("{0} : {1}".format(str(chave), str(item)))
        
        patchClf = Classificador(confs)
        patchClf.executa()
        patchClf.basetr = None
        patchClf.basets = None
        patchClf.dicttr = None
        patchClf.dictts = None
        
        classfs.append(patchClf)

    # Plota os resultados de classificacao do relatorio
    
    
    # Grava a lista de resultados para consultas posteriores
    

    # Encerramento e contagem de tempo de execucao     
    logging.info("ENCERRAMENTO DO PROGRAMA \n\n")         
            
    
# Chamada programa principal  
if __name__ == "__main__":	
	main()