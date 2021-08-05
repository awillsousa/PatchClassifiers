# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 13:28:56 2016
@author: antoniosousa

Funcoes para auxiliar no processamento dos subdiretorios e arquivos
"""
import os
#import shutil
from shutil import copy
from pathlib import Path
from fnmatch import fnmatch
import random
import re


'''
Busca arquivos em todos os subdiretorios de acordo com o padrao passado
'''
def busca_todos_arquivos(raiz, padrao="*"):
    lista = []   
    
    for caminho, subdirs, arquivos in os.walk(raiz):
        for arq in arquivos:
            if fnmatch(arq, padrao):
                lista.append(os.path.join(caminho, arq))
    return (lista)    

'''
Recebe uma lista de diretorios e de padroes e busca pelos arquivos, combinando
diretorio e padrao para cada item das duas listas. Caso, somente um padrao seja
passado ele será aplicado para todos os diretorios. Caso nenhum seja passado, recupera todos os arquivos.
'''
def busca_arquivos(diretorio, padrao="*"):            
    return (busca_todos_arquivos(diretorio, padrao))

'''
Lista todos os arquivos do diretorio corrente
'''
def lista_arquivos(caminho, padrao="*"):
    arquivos = []
    p = Path(caminho)
    arquivos = [x for x in p.iterdir() if x.is_file() and fnmatch(x, padrao)]
    return (arquivos)    
    
    
'''
Lista todos os subdiretorios do diretorio passado
'''
def lista_diretorios(caminho):
    diretorios = []
    p = Path(caminho)
    diretorios = [x for x in p.iterdir() if x.is_dir()]
    return (diretorios)
   
'''
Retorna o codigo do paciente do arquivo
'''  
def paciente_arquivo(nome_arquivo):
    info_arquivo =  str(nome_arquivo[nome_arquivo.rfind("/")+1:]).split('-')
            
    return (re.sub('[A-Z]','',info_arquivo[2]))

'''
Retorna o codigo de ampliacao do arquivo
'''
def magnif_arquivo(nome_arquivo):
    info_arquivo =  str(nome_arquivo[nome_arquivo.rfind("/")+1:]).split('-')
            
    return (info_arquivo[3])
    
'''
Gera pastas para treinamento e teste
Para cada pasta criada é feito um balanceamento da quantidade de pacientes
com condição benigna e maligna. Bem não é permitido que um paciente utilizado
na base de treino apareça na base de testes. 
'''
def gera_pastas(diretorio, mag='200', perc=0.3, num_pastas=5):

    # busca todos os arquivos da magnitude passada
    lst_imgs_ben = busca_arquivos(diretorio, "*_B_*-"+mag+"-*.png")
    lst_imgs_mal = busca_arquivos(diretorio, "*_M_*-"+mag+"-*.png")
    
    # gera uma lista de todos pacientes
    pacientes = []
    lst_arqs_pcte = []    
    lst_todas = lst_imgs_ben + lst_imgs_mal
    lst_todas = sorted(lst_todas)
    ant = paciente_arquivo(lst_todas[0])
    for l in lst_todas:
        p = paciente_arquivo(l)
        if (p != ant): # novo paciente
            #Adiciona o paciente e a sua lista de arquivos
            pacientes.append((p,lst_arqs_pcte))            
            lst_arqs_pcte = [] 
            lst_arqs_pcte.append(l)                            
        else:
            lst_arqs_pcte.append(l)
            
        ant = p
    
    # Gera um conjunto com os indices do total de pacientes
    # e gera combinações (folds) para os conjutos de treino e
    # teste
    idx_pacientes = [x for x in range(0,len(pacientes))]
    idx_pct_M = []  # lista de pacientes com tipo maligno
    idx_pct_B = []  # lista de pacientes com tipo benigno
    for idx in idx_pacientes:
        if (pacientes[idx][1][0].find("_M_") != -1):
            idx_pct_M.append(idx)
        else:
            idx_pct_B.append(idx)
        
    pastas = []    
    for i in range(num_pastas):
        random.shuffle(idx_pct_M)
        random.shuffle(idx_pct_B)
        # utiliza 70% dos pacientes do tipo maligno e
        # 70% do tipo benigno para treino
        treino = idx_pct_M[:int(len(idx_pct_M)*0.7)]
        treino += idx_pct_B[:int(len(idx_pct_B)*0.7)]
        # utiliza 30% dos pacientes do tipo maligno e
        # 30% do tipo benigno para treino
        teste = idx_pct_M[int(len(idx_pct_M)*0.7)+1:]
        teste += idx_pct_B[int(len(idx_pct_B)*0.7)+1:]
        
        pastas.append((treino, teste))
        
    return (pacientes, pastas)


'''
Cria os diretorios das pastas de testes e treinamento. 
Copiando os arquivos para o seus respectivos locais
'''
def cria_pastas(dir_base_folds, pacientes, pastas):
    # cria um diretorio para armazenar as pastas
    try:
        dir_base_folds += "FOLDS"
        os.makedirs(dir_base_folds, exist_ok=True)
    except OSError:
        pass
    
    # Itera a lista de pastas e cria as pastas
    # e copia os arquivos para cada uma das pastas
    i=0
    for p in pastas:
        #cria a pasta
        i += 1
        dir_fold = dir_base_folds + "/fold" + str(i)
        dir_treino = dir_fold + "/treino"
        dir_teste = dir_fold + "/teste"    
        
        try:
            os.makedirs(dir_fold, exist_ok=True)
            os.makedirs(dir_treino, exist_ok=True)
            os.makedirs(dir_teste, exist_ok=True)
        except OSError:
            pass
        
        # copia os arquivos de treino
        for idx_arq in p[0]:
             arquivos = pacientes[idx_arq][1]  
             for arquivo in arquivos:
                 copy(arquivo, dir_treino)
            
        # copia os arquivos de teste
        for idx_arq in p[1]:
             arquivos = pacientes[idx_arq][1]  
             for arquivo in arquivos:
                 copy(arquivo, dir_teste)
  
'''
Recebe uma lista de arquivos e embaralha eles, separando em 3 partes:
treino (65%), teste (15%) e validacao (20%)
'''
def sugere_bases(lista_arquivos):
    lista_ord = sorted(lista_arquivos)
    
    t = len(lista_ord)
    
    s = int(0.65*t)    
    p = int(0.8*t)    
    treino = lista_ord[:s-1]        
    teste  = lista_ord[s:p-1]    
    valida = lista_ord[p:]     
        
    return (treino,teste,valida)
  
def separa_bases():
    diretorio = ["/home/bases_cancer/"]        
    padrao = ["???_M_*.png"]
    lista_M = busca_arquivos(diretorio, padrao)    
    treino,teste,validacao = sugere_bases(lista_M)
    
    padrao = ["???_B_*.png"]
    lista_B = busca_arquivos(diretorio, padrao)    
    treino_B,teste_B,validacao_B = sugere_bases(lista_B)
    
    treino += treino_B
    teste  += teste_B
    validacao += validacao_B
    
    for a in treino:
        shutil.copy2(a, diretorio[0]+"treino/")
        
    for a in teste:
        shutil.copy2(a, diretorio[0]+"teste/")
    
    for a in validacao:
        shutil.copy2(a, diretorio[0]+"validacao/")

        