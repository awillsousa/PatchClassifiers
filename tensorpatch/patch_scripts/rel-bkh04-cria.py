# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 13:34:10 2016
@author: antoniosousa

Executa procedimentos do relatorio BKH04

1) Realiza extracao de atributos utilizando CNN e PFTAS, com tamanhos de patches diversos, 
diferentes valores de sobreposicao de patches

                
"""
import sys
import logging
import datetime
import ConfVars
from Extrator import Extrator
from os import path
from time import time
from optparse import OptionParser

# PROGRAMA PRINCIPAL 
def main():
    
    # Dicionario com todas as execuções
    # {'dir_imgs': "", 'dir_test': "", 'metodo': "", 'prefixo': "", 'tamanho': 64, 'sobreposicao': 100}
    dir_imgs = "/home/willian/bases/cancer/fold5/train/400X/"
    dir_dest = "/home/willian/bases/execs/cancer/fold5/train/400X/"
    metodo = "cnn"
    
    execs = [{'prefixo': "f5-400X-s128-d100", 'tamanho': 128, 'sobreposicao': 100},
             {'prefixo': "f5-400X-s128-d50",  'tamanho': 128, 'sobreposicao': 50},
             {'prefixo': "f5-400X-s128-d25",  'tamanho': 128, 'sobreposicao': 25},
             {'prefixo': "f5-400X-s64-d100",  'tamanho': 64,  'sobreposicao': 100},
             {'prefixo': "f5-400X-s64-d50",   'tamanho': 64,  'sobreposicao': 50},
             {'prefixo': "f5-400X-s64-d25",   'tamanho': 64,  'sobreposicao': 25},
             {'prefixo': "f5-400X-s32-d100",  'tamanho': 32,  'sobreposicao': 100},
             {'prefixo': "f5-400X-s32-d50",   'tamanho': 32,  'sobreposicao': 50},
             {'prefixo': "f5-400X-s32-d25",   'tamanho': 32,  'sobreposicao': 25},
             {'prefixo': "f5-400X-s16-d100",  'tamanho': 16,  'sobreposicao': 100},
             {'prefixo': "f5-400X-s16-d50",   'tamanho': 16,  'sobreposicao': 50},
             {'prefixo': "f5-400X-s16-d25",   'tamanho': 16,  'sobreposicao': 25}]
    
    for metodo in ['cnn', 'pftas']:
        for d in execs:    
            prefixo = d['prefixo']
            tamanho = d['tamanho']
            sobreposicao = d['sobreposicao']
            
            # Dicionario de configuracoes do processo de extracao
            confs = {}
            
            ## Cria a entrada de log do programa
            idarq = "bkh04-extrai"
                
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
            logging.info("Preparando parametros ...")
            
            confs['dir_imgs'] = dir_imgs    
            confs['dir_destino'] = dir_dest    
            confs['metodo'] = metodo
                
            # Verifica se os parametros foram passados corretamente
            # de acordo com o tipo de extracao a ser realizado
            confs['extracao'] = {}    
            confs['extracao']['descricao'] = "janela"                
            confs['extracao']['grava'] = False                 
            confs['extracao']['parametros'] = {}     
            confs['extracao']['parametros']['tamanho'] = tamanho
            confs['extracao']['parametros']['sobrepoe'] = sobreposicao
            
            for chave, item in confs.items():
                logging.info("{0} : {1}".format(str(chave), str(item)))
            
            patchExtrator = Extrator(confs)
            patchExtrator.executa()

    logging.info("ENCERRAMENTO DO PROGRAMA \n\n")         
            
    
# Chamada programa principal  
if __name__ == "__main__":	
	main()