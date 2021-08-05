#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 22:12:57 2017

@author: willian
"""

import sys
import logging
import pickle
import ConfVars
from Imagem import Imagem
from Patch import Patch
from Descritor import Descritor
from Extracao import Extracao
from BaseAtributos import BaseAtributos
from DictBasePatches import DictBasePatches
from pathlib import Path
from fnmatch import fnmatch
from os import path, walk


'''
Classe para extracao de atributos a partir de um conjunto de imagens 
'''
class Extrator:
    def __init__(self, conf):        
        
        if conf is None:
          sys.exit("Configuracao ausente.")
        
        self.conf = conf
        self.extracao = Extracao(self.conf['extracao'])
        self.descritor = Descritor(self.conf['metodo'])      
        
        
    # Verifica se o dicionario de configuracao possui todos os parametros
    # necessario para a execução        
    def checkConf():
        return (True, "OK")

    # Gera uma lista com todos os arquivos de imagem em um diretorio
    def listaArqs(self, diretorio, padrao):
        logging.info("Listando arquivos - diretorio: {0}".format(diretorio))
        lista = []
        for caminho, subdirs, arquivos in walk(diretorio):
            for arq in arquivos:
                if fnmatch(arq, padrao):
                    lista.append(path.join(caminho, arq))
        
        logging.info("Encontrados {0} arquivos do tipo {1}".format(len(lista), padrao))
        return (lista)

    # Executa a extracao do conjunto de patches e dos atributos
    # do conjunto extraido, utilizando os descritores especificados
    # na configuracao da classe
    def executa(self, padrao_imagem=None):
        lista_imgs = []
        
        if padrao_imagem:
           lista_imgs += self.listaArqs(self.conf['dir_imgs'], padrao_imagem)     
        else:    
            for padrao in ConfVars.TIPO_IMGS:
                lista_imgs += self.listaArqs(self.conf['dir_imgs'], padrao)
            
        if not(lista_imgs):
            sys.exit("Não há imagens para processar. Diretorio: {0}".format(self.conf['dir_imgs']))
        
        
        # Cria uma base de atributos para o descritor        
        arqbase = path.join(self.conf['dir_destino'], path.basename(self.conf['log']).replace(".log", "-") + self.descritor.tipo + ".svm")            
        base = BaseAtributos(arqbase, usa_qid=True)
                
        # Cria um dicionario para conter informacoes das bases, das imagens e dos patches        
        dict_bases_ptxs = DictBasePatches(arqbase)
        
        id_ptx = 0
        for num_img, arqimg in enumerate(lista_imgs):            
            imagem = Imagem(arqimg)
            
            logging.info("\nProcessando imagem {0}".format(arqimg))                
            ptxs_img = self.extracao.executa(imagem)
            ids_patches = []            
            for num_ptx,ptx in enumerate(ptxs_img): 
                logging.info("\tProcessando patch {0}".format(num_ptx))
                base.addAtribs(self.descritor.executa(ptx),imagem.rotulo, id_ptx)
                dict_bases_ptxs.addPatch(id_ptx, *ptx.posicao, ptx.tamanho, ptx.imagem, imagem.rotulo)
                ids_patches.append(id_ptx)
            
                # Grava os patches como imagens, se configurado 
                if self.conf['extracao']['grava']:
                    logging.info("\tGravando patch {0} para arquivo".format(num_ptx))
                    ptx.toArq(self.conf['dir_destino'])
                
                id_ptx+=1
                
            dict_bases_ptxs.addImagem(arqimg, imagem.tamanho, imagem.rotulo, ids_patches)    
        base.gravaArq()
        pickle.dump( dict_bases_ptxs, open(arqbase.replace('.svm','.ppi'), "wb" ) )
        
        
         