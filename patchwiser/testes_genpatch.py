# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 16:24:26 2016

@author: antoniosousa
Script de teste das funcoes do programa de extracao (extrai.py)

"""
import unittest 
import extrai as ext
import arquivos as arq
import logging
import GenPatch_pso as pso
import numpy as np
from time import time
import os


class TestExtraiMethods(unittest.TestCase):               
    
    def test_carrega_base(self):
        #base = "/home/willian/bases/min_treino/base_pftas_ptx64x64.svm"
        base = "Y:/bases/min_treino/base_pftas_ptx64x64.svm"
        atribs, rotulos = pso.carrega_base(base)
        
        self.assertTrue(len(rotulos)>0)
    
    def teste_generate(self):
        size = 2 
        pmin = 0
        pmax = 1
        smin = -2
        smax = 2
        
        for i in range(25):
            particula = pso.generate(size, pmin, pmax, smin, smax)        
            print("Particula {2}: {0} Velocidade: {1}".format(str(particula), particula.speed, i)) 
            
        self.assertTrue(particula != None)
        
    def teste_updateParticle(self):        
        size = 2 
        pmin = 0.35
        pmax = 0.65
        smin = -3
        smax = 3
        
        particula = pso.generate(size, pmin, pmax, smin, smax)    
        
        best = pso.generate(size, pmin, pmax, smin, smax)        
        particula.best = best
        maior = particula
        for i in range(30):            
            print("Particula {2}: {0} Velocidade: {1}".format(str(particula), particula.speed, i)) 
            pso.updateParticle(particula, best, 2.0, 2.0)
            if particula[0] > maior[0] or particula[1] > maior[1]:
                maior = particula
        
        print(str(maior))    
            
        self.assertTrue(particula != None)
        
        
# Programa principal
if __name__ == '__main__':
    
    ## Cria a entrada de log do programa
    idarq=''
    logging.basicConfig(filename='unittestes'+idarq+'.log', format='%(message)s', level=logging.INFO)    
    # Configura para o log ser exibido na tela
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)            
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)   
    logging.getLogger('').addHandler(console)   
    
    #unittest.main()
    t = TestExtraiMethods()
    #t.teste_generate()
    t.teste_updateParticle()
    