#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 22:12:57 2017

@author: willian
"""

import ConfVars
import numpy as np
import logging
from sklearn.datasets import dump_svmlight_file, load_svmlight_file



'''
Classe Base de Atributos de um conjunto de imagens 
'''
class BaseAtributos:
    def __init__(self, arquivo, tam_atribs=162, usa_qid=False, formato=ConfVars.FMT_SVMLIGHT):        
        self.formato = formato
        self.arquivo = arquivo
        self.usa_qid = usa_qid
        self.tam_atribs = tam_atribs
        self.atributos = []
        self.rotulos = []        
        self.qids = []
    
    # Acrescenta uma linha de atributos na base    
    def addAtribs(self, atribs, rotulo, qid):
        self.atributos.append(atribs)
        self.rotulos.append(rotulo)        
        self.qids.append(qid)
    
    # Acrescenta toda a base de atributos
    def addBaseAtribs(self, atribs, rotulos, qids):
        self.atributos = atribs
        self.rotulos = rotulos        
        self.qids = qids
        
    # Grava a base de atributos
    def gravaArq(self):
        qids = None
        if self.usa_qid:
            qids = self.qids
            #qids = [q for q in range(len(self.rotulos))]  # id para cada patch no arquivo de patches
        try:
            logging.info("Total de patches: {0}".format(str(len(self.atributos))))
            logging.info("Total de rotulos: {0}".format(str(len(self.rotulos))))
            logging.info("Total de qids(SMVLIGHT): {0}".format(str(len(qids))))
            
            dump_svmlight_file(self.atributos, self.rotulos, self.arquivo, query_id=qids)
        except IOError as e:
           logging.info("I/O erro({0}): {1}".format(e.errno, e.strerror))
        except Exception as e: # outras excecoes
           logging.info("Erro Inesperado:", str(e))    
    
    # Carrega a base de atributos        
    def carregaArq(self):
        logging.info("Carregando base de atributos")
        try:
            if self.usa_qid:
                self.atributos, self.rotulos, self.qids = load_svmlight_file(self.arquivo, dtype=np.float32, query_id=self.usa_qid, n_features=self.tam_atribs) 
            else:    
                self.atributos, self.rotulos = load_svmlight_file(self.arquivo, dtype=np.float32, query_id=self.usa_qid, n_features=self.tam_atribs) 
        except Exception as e:
            logging.info("Erro durante carregamento da base {0}: {1}".format(self.arquivo, str(e)))
            
    def __str__(self):
        s = ""
        for attr, value in self.__dict__.items():
            if attr in ("atributos", "rotulos", "qids"):
                value = value[0:10]
            s += "{0}: {1} \n".format(str(attr), str(value))
            
        return s