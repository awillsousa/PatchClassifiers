# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 22:54:03 2016

@author: willian

Programa para gerar imagens para a parte escrita do trabalho.
Utiliado como apoio para geração de exemplos dos métodos de pre-processamento
e extração de patches
"""

import extrator as ex
import arquivos as arq
import linecache
import numpy as np
import sys
from os import path
from os import makedirs
from skimage.filters import gaussian
from skimage.feature import local_binary_pattern
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn.feature_extraction import image
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from PIL import Image
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle

###############################################################################

def plota_grafico(dadosX, dadosY, arquivo="grafico.png", titulo="", tituloX="X", tituloY="Y", ):    
    plt.plot(dadosX, dadosY)
    
    # anota os pontos de classificacao
    for x,y in zip(dadosX,dadosY):        
        plt.annotate(r'$'+str(round(y,2))+'$',
                 xy=(x,y), xycoords='data',
                 xytext=(+10, +30), textcoords='offset points', fontsize=12,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.1"))

        plt.plot([x,x],[0,y], color ='green', linewidth=.5, linestyle="--")
        plt.plot([x,0],[y,y], color ='green', linewidth=.5, linestyle="--")
        plt.scatter([x,],[y,], 20, color ='red')
    
    # inclui titulo dos eixos e do grafico    
    plt.ylabel(tituloY)
    plt.xlabel(tituloX)    
    plt.title(titulo)
    
    # define os limites dos eixos
    plt.xlim(0.0, max(dadosX))
    max_y = max(dadosY)    
    if max_y < 100:
        max_y = 100
        
    plt.ylim(0.0, max_y)    
    plt.savefig(arquivo)    # salva como arquivo
    plt.clf()
       

############################################################################### 

def entropia_img(img,disco):
    entrop_val = entropy(img, disk(disco))
    
    return (entrop_val)


def limpa_imagem(img_cinza, exibe=False):
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


# Exibe os patches de uma imagem em subplots. Posiciona cada um dos
# subplots lado a lado na mesma sequencia de composicao da imagem 
# original    
def exibe_patches(img, tam_patch, rgb=True, map_cor=None):
    # gera os patches da imagem 
    patches = ex.cria_patches(img, tam_patch, tam_patch, rgb)
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
        
        v = (ax[i].get_xlim()[1]-ax[i].get_xlim()[0])
        nx = int((i%tam_x)*tam_patch) + int(tam_patch/2)        
        nx = int(nx/v + v/2)  
        ny = int(i/tam_x)*tam_patch + int(tam_patch/2)
        ny = int(ny/v + v/2)  
        
        ax[i].text(nx,ny,"{0}".format(i),color='r',ha='center',va='center',fontsize='18')  
        
        i += 1
    
    fig.subplots_adjust(wspace=0.1, hspace=0.1)    
    plt.show()
    
    return (patches)
    

# Exibe os patches cuja entropia esteja abaixo de um limiar determinado
def exibe_patches_entrop(img, img_entrop, tam_patch, limiar=6.0, rgb=False):     
    
    patches = ex.cria_patches(img, tam_patch, tam_patch, rgb=True)
    patches_entrop = ex.cria_patches(img_entrop, tam_patch, tam_patch,rgb=False)
    tam_x = int(img.shape[0]/tam_patch)
    tam_y = int(img.shape[1]/tam_patch)
    entrops_patches = []
    
    fig,axes = plt.subplots(tam_x,tam_y)
        
    ax = axes.ravel()        
    i = 0
    nx = 0
    ny = 0
    for i in range(patches.shape[0]):  
        patch = patches_entrop[i]
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([]) 
        ax[i].axis('off')
        
        entr_media = np.mean(patches_entrop[i])
        entrops_patches.append(entr_media)
        '''
        if entr_media > limiar:
            patch[patch != 0] = 0            
        '''    
        if (rgb):
            ax[i].imshow(patch, cmap='jet')
        else:
            ax[i].imshow(patch, cmap="gray")
        
        v = (ax[i].get_xlim()[1]-ax[i].get_xlim()[0])
        nx = int((i%tam_x)*tam_patch) + int(tam_patch/2)        
        nx = int(nx/v + v/2)  
        ny = int(i/tam_x)*tam_patch + int(tam_patch/2)
        ny = int(ny/v + v/2)  

        
        if entr_media <= limiar:    
            ax[i].text(nx,ny,'X',color='r',ha='center',va='center',fontsize='30')            
        '''    
        elif entr_media  > 5 and entr_media <= 6:    
            ax[i].text(nx,ny,'5-6',color='r',ha='center',va='center',fontsize='18')                
        elif entr_media  > 6 and entr_media <= 7:    
            ax[i].text(nx,ny,'6-7',color='r',ha='center',va='center',fontsize='18')                
        elif entr_media  > 7:    
            ax[i].text(nx,ny,'>7',color='r',ha='center',va='center',fontsize='18')                
        '''  
        
     
    return (entrops_patches)


# Calcula histogramas da entropia dos patches de todas as imagens dos diretorios indicados em DIR_IMGS
# Sera gerado um histograma para cada diretório
def histograma_entrop():
    DIR_IMGS = ['/home/willian/bases/cancer/fold1/test/40X/',
                '/home/willian/bases/cancer/fold1/test/100X/',
                '/home/willian/bases/cancer/fold1/test/200X/',
                '/home/willian/bases/cancer/fold1/test/400X/']
    TAM_PATCH = 64
    j=1
    for fold in DIR_IMGS:
        titulo = "FOLD test- " + str(fold)
        print(titulo)        
        lista_imgs = arq.busca_arquivos(fold, "*.png")
        
        i = 0 
        entrops_patches = []    
        for arq_img in lista_imgs:
        
            img_grey = cv2.imread(arq_img, cv2.IMREAD_GRAYSCALE)  # imagem original escala de cinza
            entrop_patch = entropia_img(img_grey, TAM_PATCH/2)  # matriz de entropia da imagem para area definidas por um disco    
            
            print("Imagem " + str(i))
            i += 1    
            patches_entrop = ex.cria_patches(entrop_patch, TAM_PATCH, TAM_PATCH,rgb=False)            
            entrops_patches += [np.mean(p) for p in patches_entrop]
            
        plt.hist(np.asarray(entrops_patches), bins='auto')
        plt.xlabel('Entropia')
        plt.ylabel('Qtd. Patches')
        plt.title(titulo)
        arquivo = "histo-entropia-fold-test-"+str(j)+".pdf"
        j += 1
        #plt.savefig(arquivo)
        plt.show()    
        plt.clf()
        
        
# Calcula histogramas da entropia dos patches de todas as imagens do diretorio indicado em DIR_IMGS
def descartes_entrop():    
    # Carrega as imagens de testes
    #DIR_IMGS = '/home/willian/basesML/bases_cancer/outras-imagens/' #min_treino/'
    DIR_IMGS = '/home/willian/bases/cancer/fold1/train/40X/'
    lista_imgs = arq.busca_arquivos(DIR_IMGS, "*.png")
    TAM_PATCH = 64
    TAMS_PATCHES = [8,16,24,32,64]
    
    i = 0 
    entrops = []
    entrops_patches = []
    for arq_img in lista_imgs:
    
        # calcula informacoes a nivel da imagem 
        img = cv2.imread(arq_img)  # imagem original (rgb)
        img_grey = cv2.imread(arq_img, cv2.IMREAD_GRAYSCALE)  # imagem original escala de cinza
        #img_limpa = limpa_imagem(img_grey) # imagem original preprocessada
        #mascara = np.where(img_limpa==255,255,0)   # mascara utilizada no preprocessamento
        #img_entrop = entropia_img(img_grey, img.shape[0]/2) # matriz de entropia da imagem     
        
        #entrop_limpa = entropia_img(img_limpa, TAM_PATCH/2)  # matriz de entropia da imagem para area definidas por um disco    
        entrop_patch = entropia_img(img_grey, TAM_PATCH/2)  # matriz de entropia da imagem para area definidas por um disco    
        
        print("Imagem " + str(i))
        i += 1    
        patches = exibe_patches(img, TAM_PATCH)            
        entrops_patches += exibe_patches_entrop(img, entrop_patch, TAM_PATCH, rgb=True)
        plt.imshow(entrop_patch, cmap=plt.cm.jet)    
        plt.show()    
        patches_entrop = ex.cria_patches(entrop_patch, TAM_PATCH, TAM_PATCH,rgb=False)
        
        entrops_patches += [np.mean(p) for p in patches_entrop]
        
    plt.hist(np.asarray(entrops_patches), bins='auto')
    plt.show()     
    

# Marca um patch em uma imagem de acordo com o identificador de patch passado
# Parametros:
#       - imagem: array (2D/3D) da imagem
#       - qid_patch: identificador do patch, contendo seu numero no arquivo de 
#                    patches por imagem                
#       - tam_patch: tamanho do patch (considerando patches quadrados)
#       - rgb: indica se a imagem é em escala de cinza ou colorida
def marca_patches_n(arq_base, qid_patches, tam_patch=64, rgb=False, dir_saida="plots/"):    
    
    if type(qid_patches) is int:
        qid_patches = [qid_patches]
    elif not type(qid_patches) is list:
        sys.exit("Esperado lista de inteiros ou inteiro.")
    
    arq_ppi = arq_base.replace(".svm",".ppi")
    print(arq_ppi)
    if path.isfile(arq_ppi):
        #linha_ppi = str(linecache.getline(arq_ppi, 1)).split(',')   # primeira linha
        linha_ppi = linecache.getline(arq_ppi, 1).replace('"','').replace('\n','').split(',')   # primeira linha
    else:
        sys.exit("Arquivo Inexistente: {0}".format(arq_ppi))

    imgs_patches = {}
    for qid_patch in qid_patches:
        ppi = int(linha_ppi[1].replace('"',''))    
        n_img = qid_patch//ppi    
        print("Numero imagem: {0}".format(n_img))
        
        linha_ppi = linecache.getline(arq_ppi, n_img+1).replace('"','').replace('\n','').split(',')   # linha da imagem passada
        print("Linha PPI: {0}".format(linha_ppi))
        arq_img = linha_ppi[-1]
            
        n_patch = qid_patch%ppi
        print("Numero Patch: {0}".format(n_patch))
        
        if not n_img in imgs_patches.keys():
            imgs_patches[n_img] = {'arquivo': arq_img, 'patches': [n_patch] }
        else:
            imgs_patches[n_img]['patches'].append(n_patch) 

    for n_img, valores in imgs_patches.items():             
        arq_img = valores['arquivo']
        n_patches = valores['patches']
        
        if path.isfile(arq_img):
            if rgb:
                img = cv2.imread(arq_img)  # imagem original (rgb)
            else:
                img = cv2.imread(arq_img, cv2.IMREAD_GRAYSCALE)  # imagem original escala de cinza
        else:
            sys.exit("Arquivo de imagem incorreto - {0}".format(arq_img))
            
        patches = ex.cria_patches(img, tam_patch, tam_patch, rgb)
            
        tam_x = int(img.shape[0]/tam_patch)
        tam_y = int(img.shape[1]/tam_patch)
        
        # Get current size
        fig_size = plt.rcParams["figure.figsize"]

        # Set figure width to 12 and height to 9
        fig_size[0] = 12
        fig_size[1] = 9
        plt.rcParams["figure.figsize"] = fig_size
        
        fig,axes = plt.subplots(tam_x,tam_y)
            
        ax = axes.ravel()        
        i = 0
        nx = 0
        ny = 0
        for i in range(patches.shape[0]):  
            patch = patches[i]
            ax[i].set_xticklabels([])
            ax[i].set_yticklabels([]) 
            ax[i].axis('off')
    
            #if entr_media < 5.5:
            #    patch[patch != 0] = 0            
            
            if (rgb):
                ax[i].imshow(patch)
            else:
                ax[i].imshow(patch, cmap="gray")
            
            v = (ax[i].get_xlim()[1]-ax[i].get_xlim()[0])
            nx = int((i%tam_x)*tam_patch) + int(tam_patch/2)        
            nx = int(nx/v + v/2)  
            ny = int(i/tam_x)*tam_patch + int(tam_patch/2)
            ny = int(ny/v + v/2)  
            
            if i in n_patches:    
                #ax[i].text(nx,ny,'X',color='r',ha='center',va='center',fontsize='18')                 
                #rect = Rectangle((1,1),v-3,v-3,linewidth=2,edgecolor='r',facecolor='none')
                #ax[i].add_patch(rect)
                axAxis = ax[i].axis()
                bordaX0 = axAxis[0]-0.7
                bordaY0 = axAxis[2]-0.2
                bordaX1 = (axAxis[1]-axAxis[0])+1     
                bordaY1 = (axAxis[3]-axAxis[2])+0.4                          
                rec = Rectangle((bordaX0,bordaY0),bordaX1,bordaY1,fill=False,lw=3, edgecolor='darkgreen')
                rec = ax[i].add_patch(rec)
                rec.set_clip_on(False)
            '''    
            else:
                #ax[i].text(nx,ny,str(i),color='black',ha='center',va='center',fontsize='12') 
                
                axAxis = ax[i].axis()
                bordaX0 = axAxis[0]-0.7
                bordaY0 = axAxis[2]-0.2
                bordaX1 = (axAxis[1]-axAxis[0])+1     
                bordaY1 = (axAxis[3]-axAxis[2])+0.4                          
                rec = Rectangle((bordaX0,bordaY0),bordaX1,bordaY1,fill=False,lw=1, edgecolor='r')
                rec = ax[i].add_patch(rec)
                rec.set_clip_on(False)
            '''
        fig.subplots_adjust(wspace=0.05, hspace=0.05)        
        dir_imagem = path.dirname(arq_img)
        nome_imagem = path.basename(arq_img)
        dir_saida = "plots/" + "/".join(dir_imagem.split('/')[-3:])+"/"
        if not path.isdir(dir_saida):
            makedirs(dir_saida)
            
        img_saida = dir_saida + nome_imagem
        plt.savefig(img_saida)
        plt.show()
        
        #return (entrops_patches)    


def cria_patch(path_imagem, qid_patch, path_dst):
    
    #nome_imagem = path_imagem.split('/')[-1]
    #dir_imagem = "".join(path_imagem.split('/')[:-2])
    nome_imagem = path.basename(path_imagem)
    dir_imagem = path.dirname(path_imagem)    
    
    print ("Nome imagem: {0}".format(nome_imagem))
    print ("Path imagem: {0}".format(dir_imagem))
    
    img = cv2.imread(path_imagem)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h_fig, w_fig, k = img.shape
    patches_img = ex.cria_patches(img, 64,64,rgb=True)

    for ptx in patches_img:
        plt.imshow(ptx)
        plt.show()
        plt.clf()
    
## PROGRAMA PRINCIPAL

IMAGENS = ["/home/willian/bases/cancer/fold1/train/400X/SOB_M_DC-14-11031-400-013.png",
           "/home/willian/bases/cancer/fold1/train/400X/SOB_B_A-14-22549CD-400-029.png",           
           "/home/willian/bases/cancer/fold1/train/400X/SOB_B_A-14-22549AB-400-007.png"]

tam_patch = 64

'''
# Marcar um patch especifico em uma imagem
arq_base = "/home/willian/bases/execs/cancer/fold1/train/400X/base_pftas_ptx64x64.svm"
qid_patch = [74969, 74968, 74967, 74950, 20055, 20066, 10, 20, 30, 40, 50, 60]
marca_patches_n(arq_base, qid_patch, tam_patch=64, rgb=True)
'''


# Exibir patches com entropia maior que um limiar
for arq_img in IMAGENS:
    img = cv2.imread(arq_img)
    img_gs = cv2.imread(arq_img, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img)
    patches_entrop = exibe_patches_entrop(img, entropia_img(img_gs, tam_patch/2), tam_patch, limiar=5.0, rgb=True)
    


'''
# Exibir os patches de uma imagem
arq_img = "/home/willian/bases/cancer/fold1/train/100X/SOB_M_PC-15-190EF-100-016.png"
img = cv2.imread(arq_img)
img_gs = cv2.imread(arq_img, cv2.IMREAD_GRAYSCALE)
patches = exibe_patches(img, tam_patch, rgb=True, map_cor=None)
plt.imshow(img)
'''


