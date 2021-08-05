#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 01:46:06 2017

@author: willian

Testes de comparacao e eliminacao de patches por similaridade
"""

import cv2
import gc
import numpy as np
import mahotas as mh
from Imagem import Imagem
from Extracao import Extracao
from fnmatch import fnmatch                                                                
from os import path, walk     
import sliding_window as sw
from matplotlib import pyplot as plt
from skimage.filters import gaussian
from skimage.feature import local_binary_pattern
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def listaArqs(diretorio, padrao):                                                                                                                                          
    lista = []                                   
    for caminho, subdirs, arquivos in walk(diretorio):
        for arq in arquivos:
            if fnmatch(arq, padrao):
                lista.append(path.join(caminho, arq))
    
    return (lista)  

def cria_patches(imagem, tamX, tamY, rgb=False):
    imagem = np.asarray(imagem)     
    window_size = (tamX,tamY) if not(rgb) else (tamX,tamY,3)
    windows = sw.sliding_window_nd(imagem, window_size)  
    
    return (windows)

def limpa_imagem(img_cinza):
    #binariza a imagem em escala de cinza
    img_bin_cinza = np.where(img_cinza < np.mean(img_cinza), 0, 255)
    
    # aplica lbp sobre a imagem em escala de cinza
    # lbp foi aplicado para evitar perda de informacao em regioes
    # proximas as regioes escuras (provaveis celulas)
    lbp_img = local_binary_pattern(img_cinza, 24, 3, method='uniform')
    
    # aplica efeito de blurring sobre a imagem resultante do lbp 
    blur_img = gaussian(lbp_img,sigma=6)    
    img_bin_blur = np.where(blur_img < np.mean(blur_img), 0, 255)
     
    # junta as duas regiões definidas pela binarizacao da imagem em escala
    # de cinza e a binarizacao do blurring    
    mascara = np.copy(img_bin_cinza)    
    for (a,b), valor in np.ndenumerate(img_bin_blur):
        if valor == 0:        
            mascara[a][b] = 0 
            
    # aplica a mascara obtida sobre a imagem original (em escala de cinza)
    # para delimitar melhor as regiões que não fornecerao informacoes (regioes
    # totalmente brancas)
    img_limpa = np.copy(img_cinza)
    for (a,b), valor in np.ndenumerate(mascara):
        if valor == 255:
            img_limpa[a][b] = 255

    return (img_limpa)

def exibe_patches(img, tam_patch, rgb=True, map_cor='gray'):
    # gera os patches da imagem 
    patches = cria_patches(img, tam_patch, tam_patch, rgb)
    tam_x = int(img.shape[0]/tam_patch)
    tam_y = int(img.shape[1]/tam_patch)
        
    fig,axes = plt.subplots(tam_x,tam_y)    
    ax = axes.ravel()
        
    i = 0
    for patch in patches:        
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        ax[i].axis('off')
        if (rgb):
            ax[i].imshow(patch)
        else:
            ax[i].imshow(patch, cmap=map_cor) 
        
        i += 1
    
    fig.subplots_adjust(wspace=0.1, hspace=0.1)    
    plt.show()    
    fig.clf()
    plt.close()
    del patches
    gc.collect()
    
    #return (patches)
    
def exec_surf():
    surf = cv2.xfeatures2d.SURF_create()    
    #arquivos = listaArqs("/home/willian/bases/cancer/mintreino/", "*.png")
    arquivos = listaArqs("/home/willian/bases/amostras/", "*.png")
    arquivos += listaArqs("/home/willian/bases/amostras/", "*.jpg")
    #arquivos += listaArqs("/home/willian/bases/amostras/", "*.bmp")
        
    for arq_img in arquivos:
        print("Arquivo {0}".format(arq_img))
        try:
            img = cv2.imread(arq_img,0)
            kp_img, des_img = surf.detectAndCompute(img,None)
            img2 = cv2.drawKeypoints(img,kp_img,None,(255,0,0),4)
            exibe_patches(img2, tam_patch=64, rgb=True)               
        except Exception as e:
            print(str(e))
            
def exec_sift():    
    sift = cv2.xfeatures2d.SIFT_create()
    #arquivos = listaArqs("/home/willian/bases/cancer/mintreino/", "*.png")
    arqs_cancer = listaArqs("/home/willian/bases/amostras/", "cancer*.png")
    arqs_especies = listaArqs("/home/willian/bases/amostras/", "especie*.png")
    arqs_dtd = listaArqs("/home/willian/bases/amostras/", "dtd*.jpg")
    #arquivos = listaArqs("/home/willian/bases/amostras/", "*.bmp")

    for arq_img in arqs_cancer:
        print("Arquivo {0}".format(arq_img))
        try:
            img = cv2.imread(arq_img,0)
            img = limpa_imagem(img)
            kp_img, des_img = sift.detectAndCompute(img,None)  
            img2 = cv2.drawKeypoints(img,kp_img,None,(255,0,0),4)
            exibe_patches(img2, tam_patch=64, rgb=True)  

            #norm_image = np.zeros(img.shape)            
            #norm_image = cv2.normalize(img, norm_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)              
            norm_image = (img - np.mean(img))/np.std(img)
            #kp_img, des_img = sift.detectAndCompute(norm_image,None)  
            #img3 = cv2.drawKeypoints(norm_image,kp_img,None,(255,0,0),4)
            exibe_patches(norm_image, tam_patch=64, rgb=False)
            
            del img
            del img2
            #del img3
        except Exception as e:
            print(str(e))
        
        


def exec1():
    arq_img = "/home/willian/bases/cancer/mintreino/SOB_B_A-14-22549AB-400-001.png"
    imgsimp = "/home/willian/bases/cancer/mintreino/SOB_B_A-14-22549AB-400-001.png"
    #imgsimp = "/home/willian/bases/simpsons/Treinamento/bart001.bmp"
    
    imagem = Imagem(arq_img)
    #imagem = cv2.imread(arq_img,0)          # queryImage
    #img1 = patches[4].dados
    #img2 = patches[4].dados
                        
    extrator = Extracao()
    patches = extrator.executaJD(imagem,64,1)
    
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    
    # Processa a imagem 
    imgout = None
    kp_img, des_img = sift.detectAndCompute(imagem.dados,None)
    f = cv2.drawKeypoints(imagem.dados,kp_img,imgout,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(imgout,),plt.show()
    
    # Processas os patches
    for i,p1 in enumerate(patches):
        print("Patch {0} ". format(i))
        img1 = p1.dados
        kp1, des1 = sift.detectAndCompute(img1,None)
        f = cv2.drawKeypoints(img1,kp1,imgout,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(imgout,),plt.show()
    
    '''
    for i,p1 in enumerate(patches[:-2]):
        print("Patch {0} ". format(i))
        img1 = p1.dados
        kp1, des1 = sift.detectAndCompute(img1,None)
        
        for i2,p2 in enumerate(patches[i:]):
            print("Comparando com o patch {0} ". format(i2))
            img2 = p2.dados
    
            # find the keypoints and descriptors with SIFT
            #kp1, des1 = sift.detectAndCompute(img1,None)
            kp2, des2 = sift.detectAndCompute(img2,None)
            if des2 is None:
                pass
            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)   # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params,search_params)
            matches = flann.knnMatch(des1,des2,k=2)
            
            # Need to draw only good matches, so create a mask
            matchesMask = [[0,0] for i in range(len(matches))]
            # ratio test as per Lowe's paper
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                    matchesMask[i]=[1,0]
            draw_params = dict(matchColor = (0,255,0),
                               singlePointColor = (255,0,0),
                               matchesMask = matchesMask,
                               flags = 0)
            img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
            plt.imshow(img3,),plt.show()
    '''



exec_sift()


