#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 17:00:36 2017

@author: willian
"""

import ConfVars
import logging
from os import path, walk
from fnmatch import fnmatch

'''
Classe Base de Imagens
'''
class BaseImagem:
    
    def __init__(self, caminho, tipo_imagem="*.png"):
        pass  
    
    def getClasse(self, arquivo):
        pass
        
    def listLabels(self):
        pass
    
    def listFolds(self):
        pass


class Fold():
    
    def __init__(self, descricao, padrao="*.png"):
        self.descricao = descricao
        self.pathTreino = None
        self.pathTeste = None
        self.pathValidacao = None
        self.arqsTreino = []
        self.arqsTeste = []
        self.arqsValidacao = []
        self.padrao = padrao

    # Recupera a lista dos arquivos de treino a partir do diretorio
    def setTreino(self, treino: str):
        self.pathTreino = treino
        self.arqsTreino = self.listaArqs(self.pathTreino)
    
    # Recupera a lista dos arquivos de teste a partir do diretorio    
    def setTeste(self, teste: str):
        self.pathTeste = teste
        self.arqsTeste = self.listaArqs(self.pathTeste)
        
    # Recupera a lista dos arquivos de validacao a partir do diretorio    
    def setValidacao(self, validacao: str):
        self.pathValidacao = validacao
        self.arqsValidacao = self.listaArqs(self.pathValidacao)
    
    # Recupera a lista dos arquivos de treino a partir de um arquivo         
    def treinoFromFile(self, arquivo, treino:str):
        self.pathTreino = treino
        try:
            with open(arquivo, 'r') as arq:
                self.arqsTreino = arq.readlines()
        except Exception as e:
            print("Erro durante a carga da lista de arquivos de treino: {0}".format(e))
            
        # ajuste o caminho se não estiver completo
        for arq in self.arqsTreino:
            arq = self.pathTreino + arq
            
    # Recupera a lista dos arquivos de teste a partir de um arquivo         
    def testeFromFile(self, arquivo, teste:str):
        self.pathTeste = teste
        try:
            with open(arquivo, 'r') as arq:
                self.arqsTeste = arq.readlines()
        except Exception as e:
            print("Erro durante a carga da lista de arquivos de teste: {0}".format(e))
            
        # ajuste o caminho se não estiver completo
        for arq in self.arqsTeste:
            arq = self.pathTeste + arq
            
    # Recupera a lista dos arquivos de validacao a partir de um arquivo         
    def validacaoFromFile(self, arquivo, validacao:str):    
        self.pathValidacao = validacao
        try:
            with open(arquivo, 'r') as arq:
                self.arqsValidacao = arq.readlines()
        except Exception as e:
            print("Erro durante a carga da lista de arquivos de validacao: {0}".format(e))
            
        # ajuste o caminho se não estiver completo
        for arq in self.arqsValidacao:
            arq = self.pathValidacao + arq    
    
    # Gera uma lista com todos os arquivos de imagem em um diretorio
    def listaArqs(self, diretorio):
        logging.info("Listando arquivos - diretorio: {0}".format(diretorio))
        lista = []
        for caminho, subdirs, arquivos in walk(diretorio):
            for arq in arquivos:
                if fnmatch(arq, self.padrao):
                    lista.append(path.join(caminho, arq))
        
        logging.info("Encontrados {0} arquivos do tipo {1}".format(len(lista), self.padrao))
        return (lista)
    

class BreakHisImagem(BaseImagem):
    
    def __init__(self, descricao, caminho, usa_subclasse=False):
        self.descricao = descricao
        self.caminho = caminho
        self.tipoImagem = "*.png"
        self.classes = ['M', 'B']
        self.subclasses = ['A', 'F', 'PT', 'TA', 'BC', 'DC', 'LC', 'MC', 'PC']        
        self.usaSubclasse = usa_subclasse
        
        if (self.usaSubclasse):
            self.labelsID = {n: l for n,l in enumerate(self.subclasses)} 
        else:
            self.labelsID = {n: l for n,l in enumerate(self.classes)} 
            
        self.labelinfile = True
        self.dirisclasse = False
        self.numFolds = 5
        self.magnitudes = ['400X', '200X', '100X', '40X']
        self.foldsMagnitudes = {}
        
        self.carregaFolds()
    
    # Inicializa os folds da base    
    def carregaFolds(self):        
        for mag in self.magnitudes:
            folds = []
            for n in range(1,self.numFolds+1):
                caminho_fold = self.caminho + "fold{0}/".format(n)
                fold = Fold("Mag{0}-Fold{1}".format(mag, n), caminho_fold)
                fold.setTreino(caminho_fold + "train/{0}/".format(mag))
                fold.setTeste(caminho_fold + "test/{0}/".format(mag))
                folds.append(fold)
            self.foldsMagnitudes[mag] = folds
        
        
    # Retorna a classe a subclasse da imagem 
    def getClasse(self, arquivo):
        info_arquivo =  str(arquivo[arquivo.rfind("/")+1:]).split('_')        
        
        if info_arquivo:
            classe = info_arquivo[1]            
            subclasse = info_arquivo[2].split('-')[0]
        else:
            logging.info("Problema ao recuperar o rotulo/classe da imagem.")
        
        if self.usaSubclasse:
            return ( subclasse )   
        else:
            return ( classe )
    

class DTDImagem(BaseImagem):
    
    def __init__(self, descricao, caminho, usa_subclasse=False):
        self.descricao = descricao
        self.caminho = caminho
        self.tipoImagem = "*.jpg"
        self.classes = ["banded", "blotchy", "braided", "bubbly", "bumpy", "chequered", \
                        "cobwebbed", "cracked", "crosshatched", "crystalline", "dotted", \
                        "fibrous", "flecked", "freckled", "frilly", "gauzy", "grid", "grooved",\
                        "honeycombed", "interlaced", "knitted", "lacelike", "lined", "marbled", \
                        "matted", "meshed", "paisley", "perforated", "pitted", "pleated", \
                        "polka-dotted", "porous", "potholed", "scaly", "smeared", "spiralled",\
                        "sprinkled", "stained", "stratified", "striped", "studded", "swirly", \
                        "veined", "waffled", "woven", "wrinkled", "zigzagged"]        

        self.labelsID = {n: l for n,l in enumerate(self.classes)}             
        self.labelinfile = True
        self.dirisclasse = True
        self.numFolds = 10                
        self.folds = {}
        self.carregaFolds()
    
    # Inicializa os folds da base    
    def carregaFolds(self):            
        for n in range(1,self.numFolds+1):
            caminho_fold = self.caminho
            fold = Fold("Fold{0}".format(n), caminho_fold)
            fold.treinoFromFile(caminho_fold    + "labels/train{0}.txt".format(n), caminho_fold)
            fold.testeFromFile(caminho_fold     + "labels/test{0}.txt".format(n),  caminho_fold)
            fold.validacaoFromFile(caminho_fold + "labels/val{0}.txt".format(n),   caminho_fold)                
            self.folds[n] = fold
        
    # Retorna a classe a subclasse da imagem 
    def getClasse(self, arquivo):
        classe =  arquivo.split('/')[-2]        
        
        if classe in self.classes:
            return ( classe )            
        else:
            logging.info("Problema ao recuperar o rotulo/classe da imagem.")
        
class EspeciesImagem(BaseImagem):
    
    def __init__(self, descricao, caminho, usa_subclasse=False):
        self.descricao = descricao
        self.caminho = caminho
        self.tipoImagem = "*.jpg"
        self.classes = ["Hardwood", "Softwood"]
        self.subclasses = '''
                        001 Ginkgo biloba,
                        002 Agathis becarii,
                        003 Araucaria angustifolia,
                        004 Cephalotaxus drupacea,
                        005 Cephalotaxus harringtonia,
                        006 Torreya nucifera,
                        007 Calocedrus decurrens,
                        008 Chamaecyparis formosensis,
                        009 Chamaecyparis pisifera,
                        010 Cupressus arizonica,
                        011 Cupressus lindleyi,
                        012 Fitzroya cupressoides,
                        013 Larix lariciana,
                        014 Larix leptolepis,
                        015 Larix sp,
                        016 Tetraclinis articulata,
                        017 Widdringtonia cupressoides,
                        018 Abies religiosa,
                        019 Abies vejari,
                        020 Cedrus atlantica,
                        021 Cedrus libani,
                        022 Cedrus sp,
                        023 Keteleeria fortunei,
                        024 Picea abis,
                        025 Pinus arizonica,
                        026 Pinus caribaea,
                        027 Pinus elliottii,
                        028 Pinus gregii,
                        029 Pinus maximinoi,
                        030 Pinus taeda,
                        031 Pseudotsuga macrolepsis,
                        032 Tsuga canadensis,
                        033 Tsuga sp,
                        034 Podocarpus lambertii,
                        035 Taxus baccata,
                        036 Sequoia sempervirens,
                        037 Taxodium distichum,
                        038 Ephedra californica,
                        039 Cariniana estrellensis,
                        040 Couratari sp,
                        041 Eschweilera matamata,
                        042 Eschweleira chartaceae,
                        043 Chysophyllum sp,
                        044 Micropholis guianensis,
                        045 Pouteria pachycarpa,
                        046 Copaifera trapezifolia,
                        047 Eperua falcata,
                        048 Hymenaea courbaril,
                        049 Hymenaea sp,
                        050 Schizolobium parahybum,
                        051 Pterocarpus violaceus,
                        052 Acacia tucumanensis,
                        053 Anadenanthera colubrina,
                        054 Anadenanthera peregrina,
                        055 Dalbergia jacaranda,
                        056 Dalbergia spruceana,
                        057 Dalbergia variabilis,
                        058 Dinizia excelsa,
                        059 Enterolobium cf schomburgki,
                        060 Inga sessilis,
                        061 Leucaena leucocephala,
                        062 Lonchocarpus subglaucencens,
                        063 Mimosa bimucronata,
                        064 Mimosa scabrella,
                        065 Ormosia excelsa,
                        066 Parapiptadenia rigida,
                        067 Parkia multijuga,
                        068 Piptadenia excelsa,
                        069 Pithecelobium jupunba,
                        070 Psychotria carthaginensis,
                        071 Psychotria longipes,
                        072 Tabebuia rosea alba,
                        073 Tabebuia sp,
                        074 Ligustrum lucidum,
                        075 Nectandra rigida,
                        076 Nectandra sp,
                        077 Ocotea porosa,
                        078 Percea racemosa,
                        079 Porcelia macrocarpa,
                        080 Magnolia grandiflora,
                        081 Talauma ovata,
                        082 Tibouchiana sellowiana,
                        083 Virola oleifera,
                        084 Campomanesia xanthocarpa,
                        085 Eucalyptus globulus,
                        086 Eucalyptus grandis,
                        087 Eucalyptus saligna,
                        088 Myrcia racemulosa,
                        089 Erisma uncinatum,
                        090 Qualea sp,
                        091 Vochysia laurifolia,
                        092 Grevillea robusta,
                        093 Grevilea sp,
                        094 Roupala sp,
                        095 Bagassa guianensis,
                        096 Brosimum alicastrum,
                        097 Ficus gomelleira,
                        098 Hovenia dulcis,
                        099 Rhamnus frangula,
                        100 Prunus sellowii,
                        101 Prunus serotina,
                        102 Faramea occidentalis,
                        103 Cabralea canjerana,
                        104 Carapa guianensis,
                        105 Cedrela fissilis,
                        106 Khaya iverensis,
                        107 Melia azedarach,
                        108 Swietenia macrophylla,
                        109 Balfourodendron riedelianum,
                        110 Citrus aurantium,
                        111 Fagara rhoifolia,
                        112 Simaruba amara        
                        '''.split(',')
                        
        self.subclasses = [s.strip().replace('\n', '') for s in self.subclasses]

        self.usaSubclasse = usa_subclasse
        self.labelsID = {n: l for n,l in enumerate(self.classes)}             
        self.labelinfile = False
        self.dirisclasse = True
        self.numFolds = 3                
        self.folds = {}
        self.carregaFolds()
    
    # Inicializa os folds da base    
    def carregaFolds(self):            
        for n in range(1,self.numFolds+1):
            caminho_fold = self.caminho
            fold = Fold("Fold{0}".format(n), caminho_fold)
            fold.treinoFromFile(caminho_fold    + "labels/train{0}.txt".format(n), caminho_fold)
            fold.testeFromFile(caminho_fold     + "labels/test{0}.txt".format(n),  caminho_fold)
            fold.validacaoFromFile(caminho_fold + "labels/val{0}.txt".format(n),   caminho_fold)                
            self.folds[n] = fold
        
    # Retorna a classe a subclasse da imagem 
    def getClasse(self, arquivo):
        classe =  arquivo.split('/')[-3]        
        subclasse =  arquivo.split('/')[-2]
        
        if self.usaSubclasse and subclasse in self.subclasses:
            return ( subclasse )
        elif classe in self.classes:
            return ( classe )            
        else:
            logging.info("Problema ao recuperar o rotulo/classe da imagem.")        