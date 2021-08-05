#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 02:11:20 2018

@author: willian
"""



def marca_patches_n(arq_img, id_patches, legenda="X", tam_patch=64, rgb=True, path_saida="plots/"):    
    import sys
    from os import path, makedirs
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt
    import extrator as ex
    import cv2

    if type(id_patches) is int:
        id_patches = [id_patches]
    elif not type(id_patches) is list:
        sys.exit("Esperado lista de inteiros ou inteiro.")
    
    ppi = 70        
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

        if (rgb):
            ax[i].imshow(patch)
        else:
            ax[i].imshow(patch, cmap="gray")
        
        v = (ax[i].get_xlim()[1]-ax[i].get_xlim()[0])
        nx = int((i%tam_x)*tam_patch) + int(tam_patch/2)        
        nx = int(nx/v + v/2)  
        ny = int(i/tam_x)*tam_patch + int(tam_patch/2)
        ny = int(ny/v + v/2)  
        
        if i in id_patches:
            cor_borda = 'red'            
            axAxis = ax[i].axis()
            bordaX0 = axAxis[0]-0.7
            bordaY0 = axAxis[2]-0.2
            bordaX1 = (axAxis[1]-axAxis[0])+1     
            bordaY1 = (axAxis[3]-axAxis[2])+0.4                          
            rec = Rectangle((bordaX0,bordaY0),bordaX1,bordaY1,fill=False,lw=4, edgecolor=cor_borda)
            rec = ax[i].add_patch(rec)
            #ax[i].text(nx-2,ny-1, str(legenda),color='r',ha='center',va='center',fontsize='22')                                                                     
            rec.set_clip_on(False)
        
    fig.subplots_adjust(wspace=0.05, hspace=0.05)        
    '''
    dir_imagem = path.dirname(arq_img)
    nome_imagem = path.basename(arq_img)    
    dir_saida = path_saida + "/".join(dir_imagem.split('/')[-3:])+"/"
    if not path.isdir(dir_saida):
        makedirs(dir_saida)
        
    img_saida = dir_saida + nome_imagem
    plt.savefig(img_saida)    
    plt.close('all')
    '''
    plt.show()
    
    #return (entrops_patches)    
   

def marca_patches(arq_img, id_patches, legenda="X", tam_patch=64, rgb=True, path_saida="plots/"):    
    import sys
    from os import path, makedirs
    from matplotlib.patches import Rectangle    
    import matplotlib.pyplot as plt
    import extrator as ex
    import cv2

    if type(id_patches) is int:
        id_patches = [id_patches]
    elif not type(id_patches) is list:
        sys.exit("Esperado lista de inteiros ou inteiro.")
    
    ppi = 70        
    if path.isfile(arq_img):
        if rgb:
            img = cv2.imread(arq_img)  # imagem original (rgb)
        else:
            img = cv2.imread(arq_img, cv2.IMREAD_GRAYSCALE)  # imagem original escala de cinza
    else:
        sys.exit("Arquivo de imagem incorreto - {0}".format(arq_img))
        
    patches = ex.cria_patches(img, tam_patch, tam_patch, rgb)
        
    tam_x = int(img.shape[1]/tam_patch)
    tam_y = int(img.shape[0]/tam_patch)
    print("Total de patches por eixo: ({},{})".format(tam_x, tam_y))
    # Get current size
    fig_size = plt.rcParams["figure.figsize"]

    # Set figure width to 12 and height to 9
    fig_size[0] = 12
    fig_size[1] = 9
    plt.rcParams["figure.figsize"] = fig_size
        
    # Cria figura
    fig,ax = plt.subplots(1)                
    ax.imshow(img)    
    for i in id_patches:
        cor_borda = 'red'
        posx = (i%tam_x)*tam_patch
        posy = (i//tam_x)*tam_patch               
        ax.add_patch(Rectangle((posx,posy),tam_patch,tam_patch,fill=False,lw=2, edgecolor=cor_borda))        
        
    plt.show()
    plt.clf()    
    
    
def show_patch(arq_img):
    from Imagem import Imagem
    from Patch import Patch
    from BaseAtributos import BaseAtributos
    from DictBasePatches import DictBasePatches
        
    tam = (64,64)
    img = Imagem(arq_img)
    
    posx = 0
    posy = 0
    rot =  'X'
    
    patches = []
    for i in [10,20,30,40,60]:        
        posx = (i%10)*tam[0]
        posy = (i//10)*tam[0] 
        rot =  'X'
        patches.append(Patch(i, arq_img, tam, (posx,posy), img.dados[posx:posx+tam[0]][posy:posy+tam[1]], rot)) 
        
    img.exibePatches(patches)

###############################################################################################################################

arq_img = "/home/willian/bases/min/cancer/mintreino/SOB_B_A-14-22549CD-400-014.png"
#marca_patches_n(arq_img, id_patches=[15,20,25,30,61,64,55])
#marca_patches(arq_img, id_patches=[0,11,22,33,44,32,69,70])
show_patch(arq_img)

