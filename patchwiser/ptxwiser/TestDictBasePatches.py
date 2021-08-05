#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 17:55:55 2017

@author: willian
"""

import unittest
from os import path, remove
from DictBasePatches import DictBasePatches

class DictBasePatchesTestCase(unittest.TestCase):
    
    def test_init(self):        
        dict_teste = DictBasePatches("arqbaseteste.arq", 32,32)
        self.assertTrue(isinstance(dict_teste, DictBasePatches))
    
    def test_addBase(self):    
        dict_teste = DictBasePatches("arqbaseteste.arq", 32,32)        
        dict_teste.addBase("arqbaseteste.arq")
        self.assertTrue(dict_teste.bases[-1] == "arqbaseteste.arq")
                
    def test_addImagem(self):    
        dict_teste = DictBasePatches("arqbaseteste.arq", 32,32)        
        dict_teste.addImagem("/home/willian/bases/amostras/cancer1.png", 700, 460)
        self.assertTrue(dict_teste.imagens[-1]["arquivo"] == "/home/willian/bases/amostras/cancer1.png")
        
    def test_addPatch(self):
        dict_teste = DictBasePatches("arqbaseteste.arq", 32,32)        
        dict_teste.addPatch(1, 0, 0, "/home/willian/bases/amostras/cancer1.png")
        self.assertTrue(dict_teste.patches[-1]["numero"] == 1)
        
        
if __name__ == '__main__':
    arqbase = open("arqbaseteste.arq", 'w+')
    arqbase.write("Arquivo de teste ")
    arqbase.close()
    
    unittest.main()      
    
    remove("arqbaseteste.arq")