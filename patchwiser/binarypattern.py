# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 14:25:12 2016
@author: antoniosousa

Funcoes relacionadas a execucao do LBP
"""

import numpy as np
import matplotlib.pyplot as plt
import sliding_window as sw
from skimage.filters import gaussian
from skimage.feature import local_binary_pattern
import cv2

RAIO = 1
PONTOS = 8*RAIO

'''
Converte a imagem passada para escala de cinza
'''
def rgb2cinza(img):
    return (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

'''
Binariza uma imagem de acordo com o limiar passado
'''
def binariza(img, limiar, maxval):
    return (np.where(img < limiar, 0, maxval))

'''
Cria patches de acordo com um grid onde cada celula
é um quadrado de lado igual ao tamanho passado para
a funcao. Caso as dimensões não sejam multiplos do tamanho
do patch, ignora as bordas. 
'''
def cria_patches(matriz, tam):    
    v, h = matriz.shape
    # calcula as bordas horizontais
    h_m1 = h % tam
    h_m2 = h_m1//2
    h_m1 -= h_m2
    # calcula das bordas verticais
    v_m1 = v % tam
    v_m2 = v_m1//2
    v_m1 -= v_m2
    
    # divide a imagem passada em patches
    colecao = []
    for i in range(v_m1, v - v_m2, tam):
        for j in range(h_m1, h - h_m2, tam):
            m = np.copy(matriz[i:i+tam,j:j+tam])
            colecao.append(m)            

    return (colecao)

def patches(img, tam, rgb=False):
    img = np.asarray(img)     
    window_size = (tam,tam) if not(rgb) else (tam,tam,3)
    windows = sw.sliding_window_nd(img, window_size)  
    
    return (windows)
 

'''
Aplica LBP na imagem
'''
def aplica_lbp(img):
    #lbp = local_binary_pattern(img, P=PONTOS, R=RAIO, method="uniform")
    lbp = local_binary_pattern(img, P=PONTOS, R=RAIO, method='uniform')
    
    return (np.asarray(lbp, dtype=np.uint32))

'''
Binariza a imagem em escala de cinza passada, utilizando como limiar
a media dos pixels da imagem. 
Aplica um filtro gaussiano sobre o resultado da aplicacao do lbp(24,3)
na imagem. Gerando uma nova imagem. Binariza essa nova imagem, usando como
limiar a media dos pixels. 

Adiciona as duas imagens binarizadas para gerar uma mascara a ser aplicada 
sobre a imagem original. Aplica essa mascara sobre a imagem original e devolve
a imagem limpa.
'''
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

'''
Calcula o histograma de imagem 
'''
def histograma(img, normaliza=True):
    n_bins = 10#img.max() + 1
    hist, _ = np.histogram(img, bins=n_bins, range=(0, n_bins), normed=normaliza)
    #img = np.array(img, dtype=np.float32)
    #hist = cv2.calcHist([img],[0],None,[256],[0,256])
    #cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
    
    return (hist) 

def plota_histograma(hist, bins):
    plt.bar(bins[:-1], hist, width = 1)
    plt.xlim(min(bins), max(bins))
    plt.show()   

'''
Exibe uma lista de imagens passadas
'''    
def mostra_imagens(imagens, tam_patch=64):
    """Display a list of images"""
    n_imgs = len(imagens)
       
    fig = plt.figure()
    
    n = 1
    for img in imagens:
        imagem = img[0]
        titulo = img[1]       

        #####################################
        v, h, _ = imagem.shape
        # calcula as bordas horizontais
        h_m1 = h % tam_patch
        h_m2 = h_m1//2
        h_m1 -= h_m2
        # calcula das bordas verticais
        v_m1 = v % tam_patch
        v_m2 = v_m1//2
        v_m1 -= v_m2
        #####################################            
        
        a = fig.add_subplot(1,n_imgs,n) 
        a.set_xticks(np.arange(0+h_m1, 700-h_m2, tam_patch))
        a.set_yticks(np.arange(0+v_m1, 460-v_m2, tam_patch))
        a.grid(True)
        if imagem.ndim == 2: 
            plt.gray() 
            
        plt.imshow(imagem)
        a.set_title(titulo)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_imgs)
    plt.show()

'''
Exibe uma lista de imagens passadas
'''    
def mostra_patches(patches):
    """Display a list of images"""
    n_imgs = len(patches)
       
    fig = plt.figure()
    
    n = 1
    for img in patches:
        imagem = img[0]
        titulo = img[1]       
        a = fig.add_subplot(1,n_imgs,n) 
        #a.set_xticks(np.arange(0, 700, tam_patch))
        #a.set_yticks(np.arange(0, 460, tam_patch))
        #a.grid(True)
        if imagem.ndim == 2: 
            plt.gray() 
            
        plt.imshow(imagem)
        a.set_title(titulo)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_imgs)
    plt.show()

'''
Calcula a entropia relativa entre dois histogramas
'''
def entropia_relativa(histA, histB):
        
    histA = np.asarray(histA, dtype=np.float)
    histB = np.asarray(histB, dtype=np.float)
    
    filt = np.logical_and(histA != 0, histB != 0)
   
    return (np.sum(histA[filt] * np.log2(histA[filt] / histB[filt])))
    
'''
Calcula a distancia chi quadrado entre os dois histogramas
'''    
def custom_chi2_dist(self, histA, histB, eps = 1e-10):
	d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])
 	
	return (d)

   
'''
Calcula o valor da entropia do histograma passado
Dados fora do intervalo [media-3*desvio_padrao, media+3*desvio_padrao]
não são considerados
'''
def entropy_bin(data, width):
    
    upper = np.mean(data) + 3*np.std(data)
    lower = np.mean(data) - 3*np.std(data)
    bins = np.arange(lower, upper, step=width)

    bin_widths = bins[1:] - bins[0:-1]
    counts, bins = np.histogram(data, bins)
    p = counts / np.sum(counts)
    
    # ignore zero entries, and we can't forget to apply the analytic correction
    # for the bin width!
    entropy = -np.sum(np.compress(p != 0, p*(np.log(p) - np.log(bin_widths))))
    
    return (entropy)
    
'''
Retorna a distancia entre dois histogramas
'''
def distancia_histograma(histA, histB):
    return (cv2.compareHist(np.array(histA, dtype=np.float32),np.array(histB, dtype=np.float32),1))