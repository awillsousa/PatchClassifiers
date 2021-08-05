#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 17:55:55 2017

@author: willian
"""

import unittest
from BaseImagem import Fold, BreakHisImagem, DTDImagem, EspeciesImagem

class FoldTestCase(unittest.TestCase):
    
    def test_fold(self):
        f = Fold("Fold_X")
        
        self.assertFalse(f is None)

    def test_arquivos(self):
        f = Fold("Fold_X")
        f.setTreino("/home/willian/bases/cancer/fold1/train/400X/")
        f.setTeste("/home/willian/bases/cancer/fold1/test/400X/")
        f.setValidacao("/home/willian/bases/cancer/fold1/test/400X/")        
        
        self.assertTrue(len(f.arqsTreino) > 0)
        self.assertTrue(len(f.arqsTeste) > 0)
        self.assertTrue(len(f.arqsValidacao) > 0)
        
        
        
    def test_loadfile(self):
        f = Fold("Fold_X")
        dtd_images = "/home/willian/bases/dtd/images"
        f.treinoFromFile("/home/willian/bases/dtd/labels/train1.txt", dtd_images)
        f.testeFromFile("/home/willian/bases/dtd/labels/test1.txt", dtd_images)
        f.validacaoFromFile("/home/willian/bases/dtd/labels/val1.txt", dtd_images)        
        
        self.assertTrue(len(f.arqsTreino) > 0)
        self.assertTrue(len(f.arqsTeste) > 0)
        self.assertTrue(len(f.arqsValidacao) > 0)
        

class BreakHisImagemTestCase(unittest.TestCase):
    def test_base(self):
        base = BreakHisImagem("BreakHis-Teste", "/home/willian/bases/cancer/")
        
        self.assertEqual(base.numFolds, len(base.foldsMagnitudes['400X']))
        
    
    def test_getClasse(self):
        arquivo = "/home/willian/bases/cancer/fold1/test/400X/SOB_M_PC-14-9146-400-024.png"
        
        base = BreakHisImagem("400X", "/home/willian/bases/cancer/")        
        
        self.assertEqual(base.getClasse(arquivo), "M")

class DTDImagemTestCase(unittest.TestCase):
    def test_base(self):
        base = DTDImagem("DTD-Teste", "/home/willian/bases/dtd/")
        
        self.assertEqual(base.numFolds, len(base.folds))
    
    def test_getClasse(self):
        arquivo = "/home/willian/bases/dtd/images/banded/banded_0009.jpg"
        
        base = DTDImagem("DTD-Teste", "/home/willian/bases/dtd/")       
        
        self.assertEqual(base.getClasse(arquivo), "banded")        
    

class EspeciesImagemTestCase(unittest.TestCase):
    def test_base(self):
        base = EspeciesImagem("Especies-Teste", "/home/willian/bases/especies/")        
        self.assertEqual(base.numFolds, len(base.folds))
        
    def test_getClasse(self):
        arquivo = "/home/willian/bases/especies/Hardwood/112 Simaruba amara/11201.png"        
        base = EspeciesImagem("Especies-Teste", "/home/willian/bases/especies/")               
        
        self.assertEqual(base.getClasse(arquivo), "Hardwood")
        base.usaSubclasse = True
        print(str(base.subclasses))
        self.assertEqual(base.getClasse(arquivo), "112 Simaruba amara")
    
        
if __name__ == '__main__':    
    unittest.main()      
