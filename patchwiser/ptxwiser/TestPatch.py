#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 22:12:57 2017

@author: willian

TODO: verificar se esse arquivo é utilizado. Senão, apagar!
"""


import unittest
from os import path


class PatchTestCase(unittest.TestCase):
    
    
    def test_toArq(self):
        caminho = "/home/willian/"
        arq_img_orig = "arquivo1.png"
        num = 10
        nome = ('%s/%s-%03d.png' % (caminho, arq_img_orig.split('.')[0], num))
        
        
        self.assertTrue(path.isfile(nome))
        
    def test_getTamanho(self):
        return (self.tamanho)
    
    def test_isRGB(self):
        return (self.rgb)
    

if __name__ == '__main__':
    unittest.main()    