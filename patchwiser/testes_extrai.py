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
from time import time
import os


class TestExtraiMethods(unittest.TestCase):               
    def test_extrai_patches(self):
        diretorio = "./testes/min_treino/"
        lista_imagens = arq.busca_arquivos(diretorio, "*.png")[:2]         
        metodo = "pftas"
        vl_patch = 64
        tipo_patch = "fixo"
        prefixo="teste_"
        
        arq_atrib, arq_ppi = ext.extrai_patches(lista_imagens, diretorio, metodo, vl_patch, tipo_patch, prefixo=prefixo)
        
        if arq_atrib == None:
            print("Erro na geracao do arquivo!")
        else:
            print(arq_atrib)
            print(arq_ppi)
            self.assertTrue(os.path.isfile(arq_atrib))
        
    '''   
    def test_divide4(self):
        self.assertEqual(ext.divide4(0,0,0,0,k=4), (0,0))
        self.assertEqual(ext.divide4(128,64,0,0,k=4), (8,4))
        self.assertEqual(ext.divide4(128,128,0,0,k=4), (8,8))
    

    def exibe_cria_patches(diretorio, n_divs=3):
        lista_imagens = arq.busca_arquivos(diretorio, "*.png") 
        #converte para escala de cinza 
        for arquivo in lista_imagens: 
            img = mh.imread(arquivo) 
            img_cinza = cv2.imread(arquivo, cv2.IMREAD_GRAYSCALE) 
            
            # exibe a imagem original    
            plt.imshow(img)        
            patches = ex.cria_patches(img, 32, rgb=True)
            print("Total de patches: %f", len(patches))
            print("Tamanho do patch: %i", patches[0].shape)
                        
            exibe_patches(patches, rgb)   

    def exibe_patches(patches, rgb=False):
        try:    
            y = int(sqrt(len(patches)))
            x = y
            
            if y*x < len(patches):
                y += 1
                
            print("Eixos exibe_patches: x-"+str(x)+" y-"+str(y))
            fig,axes = plt.subplots(x,y) 
             
            for i, ax in enumerate(axes.ravel()): 
                if (i == len(patches)):
                    break;
                ax.xaxis.set_major_formatter(plt.NullFormatter())
                ax.yaxis.set_major_formatter(plt.NullFormatter())
                if rgb:
                    im = ax.imshow(patches[i])
                else:
                    im = ax.imshow(patches[i],'gray')         
            plt.show()
        except Exception as e:
            print("Erro <exibe_patches>: "+str(e))
    
    '''
        
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
    
    unittest.main()
    
    '''
    diretorio = "./testes/min_treino/"
    lista_imagens = arq.busca_arquivos(diretorio, "*.png")[:2]         
    metodo = "pftas"
    vl_patch = 64
    tipo_patch = "fixo"
    prefixo="teste_"
    
    arq_atrib, arq_ppi = ext.extrai_patches(lista_imagens, diretorio, metodo, vl_patch, tipo_patch, prefixo=prefixo)
    
    if arq_atrib == None:
        print("Erro na geracao do arquivo!")
    else:
        print(arq_atrib)
        print(arq_ppi)
    '''