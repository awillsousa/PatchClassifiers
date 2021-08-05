# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:17:46 2016

@author: antoniosousa
"""

import numpy as np
import mahotas as mh
import binarypattern as bp
import sliding_window as sw
import cv2
#import arquivos as arq
from math import floor
import sys
from skimage.filters import gaussian
from skimage.feature import local_binary_pattern 
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import img_as_float
from datetime import datetime
# utilizado apenas para debugar - Hermanoteu, apagai-lo quando nao mais o precisar 
import testes_extrai as testes

# Constantes
CLASSES = {'B':0, 'M':1}    # classes das imagens de cancer { B - benigno, M - maligno }
SUBCLASSES = {'A':0, 'F':1, 'TA':2, 'PT':3, 'DC':4, 'LC':5, 'MC':6, 'PC':7}  # subclasses das imagens de cancer
RAIO = 3         # raio padrao a considerar para o LBP
PONTOS = 24      # qtd padrao de pontos a considerar para o LBP
TAM_PATCH = 64   # tamanho padrao do patch fixo   
INCR_PATCH = 4   # incremento de tamanhos de patches
LIMIAR_DIVERGENCIA=20.0     # limiar de divergencia entre os histogramas de um patch e do patch de referencia
N_DIV = 5                  # qtd padrao de divisoes a serem efetuadas na obtenção de patches 
DISCO_ENTROPIA = 32
LIMIAR_ENTROPIA = 6.5

GRAVA=False
#CAMINHO_PATCHES="/home/willian/basesML/bases_cancer/folds-spanhol/patches/"
CAMINHO_PATCHES="/home/willian/basesML/bases_cancer/patches/min/"

'''
Cria patches de tamanho tamX,tamY utilizando o metodo de janela
deslizante sem sobreposição
'''    
def cria_patches(imagem, tamX, tamY, rgb=False):
    imagem = np.asarray(imagem)     
    window_size = (tamX,tamY) if not(rgb) else (tamX,tamY,3)
    windows = sw.sliding_window_nd(imagem, window_size)  
    
    return (windows)

'''
Recebe uma area e divide a mesma k vezes
em 4 partes iguais. Retorna o tamanho da menor divisao 
(0,0,x,y)
'''
def tam4_patch(x,y,x0=0,y0=0,k=1):
    if (k == 0):
        return x,y
        
    # calcula os valores medios
    y_m = int(floor((y - y0)/2))             
    x_m = int(floor((x - x0)/2))
    k -= 1
    
    return (tam4_patch(x_m, y_m, x0, y0, k))    

'''
Extrai n_div**4 patches da imagem passada utilizando o metodo de 
janela deslizante, sem sobreposição.
'''
def cria_patches_quad(imagem, n, rgb=False):     
    if (rgb):
        l,h,_ = imagem.shape
    else:    
        l, h = imagem.shape        
    tamX, tamY = tam4_patch(l,h,0,0,n)   
     
    return (cria_patches(imagem, tamX, tamY, rgb))

def patches_img(imagem, n, tipo, rgb=False):       
    if tipo=="fixo":
       return (cria_patches(imagem, n, n, rgb))
    elif tipo=="dinamico":   
        if (rgb):
            l,h,_ = imagem.shape
        else:    
            l, h = imagem.shape        
        tamX, tamY = tam4_patch(l,h,0,0,n)   
         
        return (cria_patches(imagem, tamX, tamY, rgb))
    
'''
Extrai patches de tamanho fixo da imagem passada utilizando o metodo de
janela deslizante. 
'''
def cria_patches_fixo(imagem, lado, rgb=False):          
    return (cria_patches(imagem, lado, lado, rgb))

def patch_referencia(tam_patch):
    patch_ref = np.full([tam_patch,tam_patch], 255, dtype=np.uint8)    
        
    return (patch_ref)

def hist_referencia(patch):
    ref = patch_referencia(patch.shape[0])
    hist_ref = bp.histograma(bp.aplica_lbp(ref)) 
    
    return (hist_ref)    

'''
Definir se o patch é valido ou não para uso. 
Foi substituida pela função descarte_histograma
'''
def patch_valido(hist, hist_ref):    
    r = bp.distancia_histograma(hist, hist_ref)           
    if (r > LIMIAR_DIVERGENCIA):
        return (True) 
    
    return(False)
  
    
'''
Avalia o descarte de patch baseado em distancia de histograma
'''
def descarte_histograma(patch):    
    if not hasattr(descarte_histograma, "hist_ref"): 
        descarte_histograma.hist_ref=hist_referencia(patch)
    
    lbp_patch = bp.aplica_lbp(patch)
    hist = bp.histograma(lbp_patch)                  
    dist = bp.distancia_histograma(hist, descarte_histograma.hist_ref)  
    print("Distancia: " + str(dist))    
    if (dist > LIMIAR_DIVERGENCIA):
        return False;
    
    return True;
    
'''
Avalia o descarte de patch baseado em entropia
'''    
def descarte_entropia(patch):        
    media = np.mean(patch)    
    if (media > LIMIAR_ENTROPIA):
       return False;
    
    return True;      

'''
Avalia o descarte de patch baseado em complexidade fractal
'''    
def descarte_fractal():
    return True;

'''
Avalia o descarte de patch baseado em maximizacao da esperanca
'''
def descarte_maxes():
    return True;    


'''
Avalia se o patch deverá ser descartado ou nao
Utilizando para isso metodos de avaliar se o patch
é ou não representativo:
histo - histograma
entro - entropia
fract - complexidade fractal
'''
def descarta_patch(patch, metodo, rgb=False):
    descarta = False    
    if metodo == "histo":   # usa distancia de histograma (qi-quadrado)
       descarta = descarte_histograma(patch) 
    elif metodo == "entro": # usa entropia
        descarta = descarte_entropia(patch)
    elif metodo == "fract": # usa complexidade fractal
        descarta = descarte_fractal()
    elif metodo == "maxes": # maximizacao da esperanca
        descarta = descarte_maxes()
    else:
        sys.exit("Metodo de descarte de patches desconhecido.")
    
    return (descarta)

'''
Extrai atributos da imagem passada
img - matriz contendo informacoes da imagem
classe - rotulo da classe (que sera o mesmo dos patches)
n -  indica a quantidade de divisoes a executar para obter o 
     tamanho dos patches a serem gerados
     ou tamanho quadrado fixo nxn dos patches
tipo_patch - se a forma de gerar os patches será dinamica ou fixa
metodo - extrator de caracteristicas a ser utilizado (haralick, pftas, lbp, etc)
'''
def extrai_atrib_patches(img, classe, n, tipo_patch, metodo, descarta=False):
    try:
        ##print("Tamanho da imagem passada: {0}".format(img.shape))
        ##print("Parametros: classe - {0} / n - {1} / tipo_patch - {2} / metodo - {3} / descarta - {4}".format(classe, n, tipo_patch, metodo, descarta))
        patches = patches_img(img, n, tipo_patch, rgb=True)
        ##print("Quantidade de patches gerados: {0}".format(patches.shape))
        # gera os patches para o processo de descarte            
        if descarta:
            img_gs = mh.colors.rgb2gray(img)#, dtype=np.int8)
            img_entrop = entropy(mh.colors.rgb2gray(img, dtype=np.uint8), disk(32)) 
            img_gs = limpa_imagem(img_gs)
            
            patches_gs = patches_img(img_gs, n, tipo_patch, rgb=False)                        
            patches_entrop = patches_img(img_entrop, n, tipo_patch, rgb=False)
           
        #testes.exibe_patches(patches)   #apenas para debugar
        #testes.exibe_patches(patches_gs)   #apenas para debugar
                
        ppor_img = len(patches)   # total de patches por img
        pdesc_img = 0   # total de patches a serem descartados
        
        atributos = []
        rotulos = []
        descartados = []  #apenas para debugar
        #testes.exibe_patches(patches)
        for i,patch in enumerate(patches):
            if descarta and descarta_patch(patches_entrop[i],metodo="entro",rgb=True):
                pdesc_img += 1
                descartados.append(patches_gs[i])#apenas para debugar
            else:
                if GRAVA:
                    idarq=datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
                    nfname = ('%s/%s-%03d%s.png' % (CAMINHO_PATCHES, "patch", i,idarq))
                    cv2.imwrite(nfname, patch)
                
                atrs,rots = atributos_img(patch, classe, metodo)
                atributos.append(atrs)
                rotulos.append(rots)
        #print("Qtd descartados: " + str(pdesc_img))     #apenas para debugar
        #if len(descartados) > 1:        #apenas para debugar
        #    testes.exibe_patches(descartados) 
            
        return (atributos, rotulos, (ppor_img,pdesc_img))      
    except Exception as e:
        print("Erro <extrai_atrib_patches>: " + str(e))


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

def classe_arquivo(nome_arquivo):
    info_arquivo =  str(nome_arquivo[nome_arquivo.rfind("/")+1:]).split('_')        
    classe = info_arquivo[1]

    # utilizado para executar o mesmo codigo da base de cancer para
    # uma outra base de teste. Eliminar da versao final (GO HORSE!!!)    
    if len(info_arquivo) <= 2:
        return ("B","X")
        
    subclasse = info_arquivo[2].split('-')[0]
    
    return (classe,subclasse)


'''
Extrai descritores PFTAS do patch
'''    
def extrai_pftas(patch):
    # extrai os atributos de cada um dos patches
    return (mh.features.pftas(patch))

'''
Extrai descritores LBP do patch
'''  
def extrai_lbp(img):
    im = mh.colors.rgb2grey(img)
    return (mh.features.lbp(im, radius=3, points=24))
    #return(bp.aplica_lbp(im))

'''
Extrai descritores GLCM(Haralick) do patch
haralick_labels = ["0 - Angular Second Moment (Uniformity) Energy = (Uniformity)^1/2",
                   "1 - Contrast",
                   "2 - Correlation",
                   "3 - Sum of Squares: Variance (Contrast)",
                   "4 - Inverse Difference Moment (Local Homogeneity)",
                   "5 - Sum Average",
                   "6 - Sum Variance",
                   "7 - Sum Entropy",
                   "8 - Entropy",
                   "9 - Difference Variance",
                   "10 - Difference Entropy",
                   "11 - Information Measure of Correlation 1",
                   "12 - Information Measure of Correlation 2",
                   "13 - Maximal Correlation Coefficient"]
'''      
def extrai_haralick(img):        
    glcm_medias = mh.features.haralick(mh.colors.rgb2gray(img, dtype=np.uint8), return_mean=True)    
    return (glcm_medias)

def atributos_img(img, classe, extrator):   
    # extrai os atributos do patches    
    if (extrator == 'pftas'):
        atributos = extrai_pftas(img)    
    if (extrator == 'lbp'):
        atributos = extrai_lbp(img)
    if (extrator == 'glcm'):
        atributos = extrai_haralick(img)           
    rotulo = CLASSES[classe]

    return (atributos,rotulo)
    
