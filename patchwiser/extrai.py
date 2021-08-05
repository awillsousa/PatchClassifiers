# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 13:34:10 2016
@author: antoniosousa
Procedimentos de extracao de caracteristicas de imagens, utilizando como refereência
métodos de extração de patches.

O programa extrai patches de duas maneiras distintas:
1a) A extração é feita dividindo a imagem em quatro partes. 
    Ao chamar o programa podem ser passados duas opções:
    a) n - para inidicar a quantidade de divisões em quatro partes que
           será executada na imagem. Por exemplo:
           n=0 equivale a imagem inteira
           n=1 divide a imagem em quatro patches de mesmo tamanho
           n=2 divide a imagem em 16 patches de mesmo tamanho
       a - para indicar a quantidade maxima de divisões que será consideradas
           Por exemplo:
           a=2, irá extrar atributos para os patches gerados por 1 divisão 
                em quatro partes, gravar em arquivo, e a seguir divide a imagem
                duas vezes em quatro partes, extrair os atributos de cada patch
                e gerar um novo arquivo da base.
           Ao se usar essa opção, será gerado um arquivo de base para cada "nível"
           de divisão a ser executado.
       s - para indicar patches quadrados de tamanho fixo. Essa opção pode ser usada
           com apenas um tamanho de patch ou com um intervalo.
           Por exemplo: 
           s=4, irá extrair patches de tamanho 4x4 da imagem
           s=4,16, irá extrair um conjunto de patches nesse intervalo, determinado
                   pela opção "i". Ou seja, para i=4, serão extraídos patches de 
                   tamanho 4x4,8x8,12x12 e 16x16. Para cada um desses tamanhos será
                   gerado um arquivo. 
Durante a chamada ainda é possível especificar o método de extração a ser utilizado,
apesar de apenas pftas estar implementado.

Ao chamar o programa é possível indicar o uso de descarte ou não de patches, através
da opção "-d"
           
     
"""


from os import path
from time import time
from optparse import OptionParser

import sys
import csv
import helper
import datetime
import mahotas as mh
import extrator as ex
import arquivos as arq
import numpy as np
import logging
from sklearn.datasets import dump_svmlight_file
#import multiprocessing
#from functools import partial

'''
Gera o nome dos arquivos da base de treino, indicador de quantidades de patches
por imagem a ser utilizado e arquivo de log da extracao
'''
'''
def nomes_arquivos(tipo_patch, vl_patch, diretorio, metodo, descarta):
       
    arq_base = diretorio + "base_" + metodo
    if tipo_patch=="dinamico":
        arq_base += "_"+str(4**vl_patch)+"_patches"
    elif tipo_patch=="fixo":
        arq_base += "_patches"+str(vl_patch)+"x"+str(vl_patch)
    
    if (descarta): # base com descarte de patches
            arq_base += "D"           
                    
    arq_atrib = arq_base+".svm"
    arq_patches = arq_base+".ppi"
    arq_log = arq_base+"EXT.log"
    
    return (arq_atrib, arq_patches, arq_log)
'''
def nomes_arquivos(tipo_patch, vl_patch, diretorio, metodo, descarta, prefixo="base_"):
    logging.info("Gerando nomes de arquivos")    
       
    arq_base = diretorio + prefixo + metodo
    if tipo_patch=="dinamico":
        arq_base += "_"+str(4**vl_patch)+"_ptx"
    elif tipo_patch=="fixo":
        arq_base += "_ptx"+str(vl_patch)+"x"+str(vl_patch)
    
    if (descarta): # base com descarte de patches
            arq_base += "D"           
                    
    arq_atrib = arq_base+".svm"
    arq_patches = arq_base+".ppi"
    arq_log = arq_base+"EXT.log"
    
    return (arq_atrib, arq_patches, arq_log)    
    
# Execucação da extração de atributos paralelizada
# !!!  ESSA IMPLEMENTAÇÃO AINDA NÃO É DEFINITIVA  !!!
def extract_paralelo(vl_patch, tipo_patch, metodo, descarta, arq_imagem):
    logging.info("Extracao imagem: " + arq_imagem)
    classe, _ = ex.classe_arquivo(arq_imagem)
    imagem = mh.imread(arq_imagem)
    
    # Extrai os atributos   
    atrs,rots,tp_patch = ex.extrai_atrib_patches(imagem, classe, vl_patch, tipo_patch, metodo, descarta)    
    #atributos += atrs 
    #rotulos += rots 
    
    return (atrs,rots,tp_patch,classe,arq_imagem)
     

'''
Gera os patches da lista de imagens passada e extrai os atributos
imagens - lista de caminhos dos arquivos das imagens a serem processadas
diretorio - diretorio onde as imagens estão localizadas
metodo - o metodo de extracao de atributos que deverá ser utilizado
vl_patch - indica a quantidade de divisões (tipo_patch=dinamico) a serem efetuadas na geração dos
           dos patches
           ou
           o tamanho do patch quadrado a ser utilizado (tipo_patch=fixo)
'''
def extrai_patches(imagens, diretorio, metodo, vl_patch, tipo_patch, prefixo="base_", descarta=False):
    logging.info("Extracao de patches")
    t0=time()     
    
    # Cria os dicionarios dos atributos e rotulos por qtd de patches
    atributos = []      # lista de atributos extraidos dos patches
    rotulos = []        # lista de rotulos dos patches
    atributos_glcm = []      # lista de atributos 
    rotulos_glcm = []        # lista de patches
    p_por_img = []      # lista de tuplas indicando (num_patch, patches por img, patches descartados)    
    
    # Gera o nome dos arquivos da base de treino e indicador de quantidades de patches
    # por imagem a ser utilizado     
    arq_treino, arq_patches, arq_log = nomes_arquivos(tipo_patch, vl_patch, diretorio, metodo, descarta, prefixo)         
    arq_glcm = arq_treino.replace('.svm', '.glcm')

    # verifica se o arquivo da base de treino e da base de patches já existe
    # se existir retorna 
    print("Arq treino: " + arq_treino)
    print("Arq patches: " + arq_patches)
    if path.isfile(arq_treino) and path.isfile(arq_patches):
       logging.info("ARQUIVOS JÁ EXISTEM: " + arq_treino + " - " + arq_patches) 
       return(arq_treino, arq_patches)   
        
    logging.info("Processo de extracao de atributos")
    logging.info("Arquivo base: " + arq_treino)
    logging.info("Arquivo de patches: " + arq_patches)
    
    tempos_imgs = []
    
    ##  INICIO DO PROCESSO DE EXTRACAO DE ATRIBUTOS
    for i,arq_imagem in enumerate(imagens):
        logging.info("Extracao imagem: " + arq_imagem)
        t0_imagem = time()
        imagem = mh.imread(arq_imagem)
        classe, _ = ex.classe_arquivo(arq_imagem)
        
        # Extrai os atributos   
        atrs,rots,tp_patch = ex.extrai_atrib_patches(imagem, classe, vl_patch, tipo_patch, metodo, descarta)    
        atributos += atrs 
        rotulos += rots         

        #coment = [i, tp_patch[0], tp_patch[1], classe, arq_imagem]
        #str_coment = [str(x) for x in coment]
        #comentarios += ["|".join([str(id_ptx)]+str_coment) for id_ptx in range(len(atributos))]        
        #print(str(comentarios))

        # Extrai os atributos da GLCM
        atrs_glcm,rots_glcm,_ = ex.extrai_atrib_patches(imagem, classe, vl_patch, tipo_patch, "glcm", descarta)    
        atributos_glcm += atrs_glcm 
        rotulos_glcm += rots_glcm 
        
        if atrs == None or rots == None:
            logging.info("Erro nos atributos ou rotulos gerados da imagem: " + arq_imagem)
            
        if tp_patch == None:            
            logging.info("Informações de patch para a imagem: " + arq_imagem)
        else:
            # <posicao imagem>,<patches por imagem>,<patches gerados>,<patches descartados>,<classe imagem>,<caminho imagem>
            p_por_img.append((i, tp_patch[0]-tp_patch[1], tp_patch[0], tp_patch[1], classe, arq_imagem))
    
        tempos_imgs.append(round(time()-t0_imagem,3))
        logging.info("Tempo extração imagem: {0}".format(tempos_imgs[-1]))
        
    
    # Exibe estatisticas de tempo por imagem
    logging.info("Tempo medio de extracao por imagem: {0}".format(np.mean(tempos_imgs)))
    logging.info("Desvio padrao tempo de extracao por imagem: {0}".format(np.std(tempos_imgs)))
    logging.info("Tempo extração base: " + str(round(time()-t0,3)))
    t1 = time()
    
    # gera os arquivos dos patches da base de treino           
    try:
        if len(atributos) > 0:
            qids = [qid for qid in range(len(rotulos))]  # id para cada patch no arquivo de patches                        
            dump_svmlight_file(atributos, rotulos, arq_treino, query_id=qids)   
        
        # grava o arquivo de informações de patches por imagem
        with open(arq_patches, 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)  
            for tup in p_por_img:                
                wr.writerow(tup)        
                
        if len(atributos) > 0:                      
            dump_svmlight_file(atributos_glcm, rotulos, arq_glcm, query_id=qids)
        
        return (arq_treino, arq_patches)        
    except Exception as e:        
        logging.info("Erro ao gerar o arquivo da base: " + str(e))
    
    # LOG LOG LOG  
    logging.info("Tempo de gravacao arquivos: " + str(round(time()-t1,3)))

'''
Extrai atributos e gera os arquivos da base de treino 
base - diretorio das imagens a serem processadas
metodo - metodo de extracao de atributos utilizada
tipo_path - a divisão será em patches de tamanho fixo ou usando divisões
            sucessivas
valor-patch - pode indicar a quantidade de divisoes a ser utilizada ou   
              o tamanho do patch a ser utilizado   
'''
def executa_extracao(base, metodo, tipo_div, prefixo, descarta, v1, v2=None, inc=8):
    logging.info("Execução extracao")
    arqs_atrib = []    # lista de caminhos dos arquivos de atributos (bases)
    arqs_ppi = []       # lista de caminhos dos arqivos de patches por imagem
    
    lista_imgs = arq.busca_arquivos(base, "*.png")
    if len(lista_imgs) == 0:
        lista_imgs = arq.busca_arquivos(base, "*.jpg")
        
    logging.info("Extraindo atributos da base " + base + " utilizando " + metodo)
    if tipo_div == "ALTURA":        
        for n in range(0,v1):
            if (n==0):  # não descarta para a imagem inteira
               arq_atrib, arq_ppi = extrai_patches(lista_imgs, base, metodo, n, "dinamico", prefixo, False)        
            else:             
               arq_atrib, arq_ppi = extrai_patches(lista_imgs, base, metodo, n, "dinamico", prefixo, descarta)            
            
            arqs_atrib.append(arq_atrib) 
            arqs_ppi.append(arq_ppi)            
            logging.info("Altura: "+ str(n) + " para " + str(len(lista_imgs)) + " imagens")         
    elif tipo_div == "NIVEL": 
        if v1==0:
            arq_atrib, arq_ppi = extrai_patches(lista_imgs, base, metodo, v1, "dinamico", prefixo, False)
        else:
            arq_atrib, arq_ppi = extrai_patches(lista_imgs, base, metodo, v1, "dinamico", prefixo, descarta)
        
        arqs_atrib.append(arq_atrib) 
        arqs_ppi.append(arq_ppi)        
        logging.info("Nivel: "+ str(v1) + " para " + str(len(lista_imgs)) + " imagens") 
    elif tipo_div == "TAMANHO":        
        if v2 == None: v2 = v1
        for tamanho in range(v1, v2+1, inc):
            arq_atrib, arq_ppi = extrai_patches(lista_imgs, base, metodo, tamanho, "fixo", prefixo, descarta)             
            
            arqs_atrib.append(arq_atrib)
            arqs_ppi.append(arq_ppi)
            logging.info("Tamanho: "+ str(tamanho) + " para " + str(len(lista_imgs)) + " imagens")     
   
    return arqs_atrib,arqs_ppi

'''
Verifica se uma opção passada existe na lista de argumentos do parser
'''  
def existe_opt (parser, dest):
   if any (opt.dest == dest and (opt._long_opts[0] in sys.argv[1:] or opt._short_opts[0] in sys.argv[1:]) for opt in parser._get_all_options()):
      return True
   return False 

# Insere a informação do erro no log do programa e forca saida, encerrando o programa
def loga_sai(erro):
    logging.info(erro)
    sys.exit(erro) 
 
# PROGRAMA PRINCIPAL 
def main():
    t0 = time()
    
    parser = OptionParser()
    parser.add_option("-T", "--base", dest="base",
                      help="Localização dos arquivos da base de treino")
    parser.add_option("-o", "--destino", dest="destino",
                      help="Diretório de gravação do arquivo de atributos extraidos")        
    parser.add_option("-m", "--metodo", dest="opt_metodo",
                      default='pftas',
                      help="Lista de metodos de extração de atributos. [pftas, lbp, glcm]")
    parser.add_option("-p", "--prefixo", dest="opt_prefixo",
                      default='base_',
                      help="Prefixo dos arquivos a serem gerados")               
    parser.add_option("-a", "--altura", dest="opt_altura",                      
                      help="Altura da quadtree de patches geradas.")    
    parser.add_option("-n", "--nivel", dest="opt_nivel",                      
                      help="Nível da quadtree de patches a ser utilizado.")  
    parser.add_option("-s", "--tamanho", dest="opt_tamanho",                      
                      help="Tamanho do patch quadrado a ser utilizado.")                           
    parser.add_option("-i", "--incremento", dest="opt_incremento",                      
                      help="Tamanho do incremento a ser utilizado.") 
    parser.add_option("-d", action="store_true", dest="descarte",  
                      help="Indicar se descarte de patches será usado.") 
    parser.add_option("-v", action="store_true", dest="verbose", help="Exibir saida no console.")
     
    (options, args) = parser.parse_args()
    
    ## Cria a entrada de log do programa
    idarq=datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')  
    logging.basicConfig(filename='extrai-'+idarq+'.log', format='%(message)s', level=logging.INFO)
    if options.verbose:
        # Configura para o log ser exibido na tela
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)            
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)   
        logging.getLogger('').addHandler(console)    
    logging.info("INICIO DO PROGRAMA") 
    
    # verifica se a base de treino passada existe
    if existe_opt(parser, "base") and (not (path.isdir(options.base))):
       loga_sai("Erro: Caminho da base passada incorreto ou inexistente.")
       
    # verifica se o diretorio de destino existe
    if existe_opt(parser, "destino") and (not (path.isdir(options.destino))):
       loga_sai("Erro: Caminho do diretorio de destino incorreto ou inexistente.")

    # verifica se será utilizado descarte de patches
    descarta=False    
    if options.descarte:
       descarta=True  
       
    # Seta o valor do incremento dos patches de tamanho fixo    
    inc=ex.INCR_PATCH   
    if existe_opt(parser, "opt_incremento"):
       try: 
           inc = int(options.opt_incremento)    
       except ValueError:
           loga_sai("Valor de incremento inválido! Um inteiro era esperado.")
          
    # metodo de extracao de atributos a ser utilizado
    if not(existe_opt(parser, "opt_metodo")):
       loga_sai("Erro: Metodo de extração ausente.")
        
    # verifica o tipo de divisão de patches que sera utilizada    
    fl_altura = existe_opt(parser, "opt_altura")
    fl_nivel = existe_opt(parser, "opt_nivel")
    fl_tamanho = existe_opt(parser, "opt_tamanho")
    # pelo menos um tipo de divisao deve ser escolhido
    if not(fl_altura) and not(fl_nivel) and not(fl_tamanho):
       loga_sai("Erro: Pelo menos uma divisão dos patches deve ser escolhida entre tamanho, nivel da quadtree ou altura da quadtree de patches.")
    else:
        # apenas uma das opcoes deve ser passada
        if not(helper.xor_n(fl_altura,fl_nivel,fl_tamanho)):
           loga_sai("Erro: Apenas uma opção de divisão dos patches deve ser escolhida.")         
        else: # somente um tipo de divisão
           try:
               logging.info("<<<<<<<< EXECUTA EXTRAÇÃO >>>>>>>>")
               if fl_altura:  # altura (da quadtree) foi passada
                   altura = int(options.opt_altura)
                   logging.info("Executando extração para quadtree: altura = {0}".format(altura))
                   executa_extracao(options.base, options.opt_metodo, "ALTURA", options.opt_prefixo, descarta, altura)
                   
               elif fl_nivel:   # 
                   nivel = int(options.opt_nivel)
                   logging.info("Executando extração para quadtree: nivel = {0}".format(nivel))
                   executa_extracao(options.base, options.opt_metodo, "NIVEL", options.opt_prefixo, descarta, nivel)      
                   
               else:  # tamanho foi passado
                   opt_tams = options.opt_tamanho.split(',')
                   
                   if (len(opt_tams) > 1): # passado um intervalo                                              
                      logging.info("Executando extração para intervalo de tamanhos fixos: de {0}x{0} a {1}x{1} com incrementos de {2}".format(int(opt_tams[0]), int(opt_tams[1]), inc)) 
                      executa_extracao(options.base, options.opt_metodo, "TAMANHO", options.opt_prefixo, descarta, int(opt_tams[0]), int(opt_tams[1]), inc)
                      
                   else: # passado apenas um tamanho
                      logging.info("Executando extração para tamanho fixo: {0}x{0}".format(int(opt_tams[0]))) 
                      executa_extracao(options.base, options.opt_metodo, "TAMANHO", options.opt_prefixo, descarta, int(opt_tams[0]), int(opt_tams[0]), inc)
                      
           except ValueError as v:                
                loga_sai("Valor inválido, quando um inteiro era esperado. Erro: " + str(v))
                
    logging.info("Tempo total da extracao: " + str(round(time()-t0,3)))
    
    # Encerramento e contagem de tempo de execucao 
    logging.info("Tempo total de execucao da extracao: " + str(round(time()-t0,3)))
    logging.info("ENCERRAMENTO DO PROGRAMA \n\n")         
    
    
# Chamada programa principal  
if __name__ == "__main__":	
	main()