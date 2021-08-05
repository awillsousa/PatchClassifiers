#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 01:06:42 2017

Procedimentos de extracao de caracteristicas de imagens da base BreakHis

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
                
"""
import sys
import logging
import datetime
import ConfVars
from Extrator import Extrator
from os import path
from time import time
from optparse import OptionParser

# PROGRAMA PRINCIPAL 
def main():
    t0 = time()
    
    uso = "Uso: %prog"
    
    parser = OptionParser(uso)
    
    parser.add_option("-B", "--base", dest="base",
                      type='choice', action='store',
                      choices=ConfVars.BASES,
                      default=ConfVars.BASE_DEFAULT,
                      help ="Indica a base que sera utilizada")
    
    parser.add_option("-m", "--metodo", dest="metodo",
                      default=ConfVars.METODO,
                      help="Lista de metodos de extração de atributos. [pftas, lbp, glcm, cnn]")
    parser.add_option("-p", "--prefixo", dest="prefixo",
                      default='ptx',
                      help="Prefixo dos arquivos a serem gerados")
    
    parser.add_option("-t", "--tipo", dest="tipo",
                      type='choice', action='store',
                      choices=ConfVars.TIPOS,
                      default=ConfVars.TIPO,
                      help='''Tipo da extracao a ser executada. 
                      Tipos:                          
                            janela - janela deslizante. 
                                Parametros: -s <TAMANHO> -b <% SOBREPOSIÇÃO> \n 
                             ppi - patches por imagem 
                                Parametros: -q <QTD PATCHES> \n                       
                            intervalo - intervalo de patches de tamanho fixo. 
                                Parametros: -s <TAMANHO> -b <% SOBREPOSIÇÃO> -i <INTERVALO>\n 
                            quadtree - extrai utilizando um esquema de quadtree. 
                                Parametros: -a <ALTURA DA QUADTREE> -n <NIVEL DA QUADTREE>\n  
                            randomico - extrai patches randomicos. 
                                Parametros: -s <TAMANHO> -q <QTD PATCHES>''')    
    parser.add_option("-s", "--tamanho", dest="tamanho", default=ConfVars.TAM_PATCH,         
                      help="Tamanho do patch quadrado a ser utilizado.")       
    parser.add_option("-b", "--sobrepoe", dest="perc_sobrepoe", default=ConfVars.PERC_SOBREPOE, type="int",
                      help="Percentual de sobreposição dos patches.")                               
    parser.add_option("-q", "--qtde", dest="qtde_patches", default=ConfVars.PPI, type="int",
                      help="Quantidade de patches por imagem a extrair.") 
    parser.add_option("-i", "--incr", dest="incr", default=ConfVars.INCR, type="int",            
                      help="Indica o incremento no tamanho dos patches para utilizar com extração do tipo intervalo.")     
    parser.add_option("-a", "--altura", dest="altura", type="int", 
                      help="Altura da quadtree de patches geradas.")    
    parser.add_option("-n", "--nivel", dest="nivel", type="int", 
                      help="Nível da quadtree de patches a ser utilizado.")
    
    parser.add_option("-w", action="store_true", dest="grava_ptx", help="Gravar os patches extraidos para arquivo.")    
    parser.add_option("-l", "--log", dest="log", help="Arquivo de log a ser criado.")
    parser.add_option("-v", action="store_true", dest="verbose", help="Exibir saida no console.")
     
    (options, args) = parser.parse_args()
        
    # Dicionario de configuracoes do processo de extracao
    confs = {}
    
    ## Cria a entrada de log do programa
    idarq = options.log
        
    arq_log = "logs/{0}-{1}.log".format(options.prefixo, idarq)
    
    try:
        logging.basicConfig(filename=arq_log, format='%(message)s', level=logging.INFO)
    except FileNotFoundError as e:
        sys.exit("Arquivo de log ou caminho do arquivo incorreto.")
        
    confs['log'] = arq_log
    
    if options.verbose:
        # Configura para o log ser exibido na tela
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)            
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)   
        logging.getLogger('').addHandler(console)    
    logging.info("INICIO DO PROGRAMA") 
    logging.info("Processando parametros passados...")
    
    # Verifica se foi passada uma base 
    if options.base:
        
    
    # verifica se a base de treino passada existe
    if not(options.dir_imgs) or not(path.isdir(options.dir_imgs)):
       parser.error("Caminho da base passada incorreto ou inexistente.")
    
    confs['dir_imgs'] = options.dir_imgs
    
    # verifica se o diretorio de destino existe
    if options.dir_dest and (not (path.isdir(options.dir_dest))):
       parser.error("Caminho do diretorio de destino incorreto ou inexistente.")
    
    confs['dir_destino'] = options.dir_dest
      
    # metodo de extracao de atributos a ser utilizado
    if not(options.metodo):
       parser.error("Metodo de extração ausente.")
    
    if options.metodo in ConfVars.METODOS:
        confs['metodo'] = options.metodo
    else:
        parser.error("Método {0} inválido. Valores aceitos: {1}".format(options.metodo, str(ConfVars.METODOS)))
    
    # Verifica se os parametros foram passados corretamente
    # de acordo com o tipo de extracao a ser realizado
    confs['extracao'] = {}
    if options.tipo: 
        confs['extracao']['descricao'] = options.tipo
        
        if options.grava_ptx:     
            confs['extracao']['grava'] = options.grava_ptx     
        else:
            confs['extracao']['grava'] = False
                 
        confs['extracao']['parametros'] = {}     
        if options.tipo == "janela":        
            if options.tamanho:
                confs['extracao']['parametros']['tamanho'] = int(options.tamanho)
                confs['extracao']['parametros']['sobrepoe'] = int(options.perc_sobrepoe)
                     
        elif options.tipo == "ppi":                    
                confs['extracao']['parametros']['qtde_patches'] = int(options.qtde_patches)
                     
        elif options.tipo == "intervalo":
            if options.tamanho:
               opt_tams = [int(x) for x in options.tamanho.split(',')]
               
               if (len(opt_tams) == 2):                   
                   if opt_tams[0] > opt_tams[1]: # inicial > final, troca
                       opt_tams[0],opt_tams[1] = opt_tams[1], opt_tams[0]
                   
                   if (opt_tams[1] - opt_tams[0]) < int(options.incr):
                       logging.info('''*************
                                        ATENCAO! 
                                        O intervalo passado é menor que o incremento.
                                        Somente o primeiro tamanho de patches será processado.
                                       *************''')
                    
                   confs['extracao']['parametros']['tamanho_ini'] = opt_tams[0]
                   confs['extracao']['parametros']['tamanho_fim'] = opt_tams[1]
                   confs['extracao']['parametros']['incr'] = int(options.incr)
               else:
                   parser.error("Esperado um intervalo de tamanhos.") 
        elif options.tipo == "quadtree":        
            if options.altura and options.nivel:
                parser.error("Deve ser passada apenas altura ou nivel")
            elif options.altura:
                confs['extracao']['parametros']['altura'] = options.altura
            elif options.nivel:
                confs['extracao']['parametros']['nivel'] = options.nivel
            else:
                parser.error("Para o tipo de extracao quadtree deve ser passado altura ou nivel")
                
        elif options.tipo == "randomico":        
            if options.tamanho:
                try:
                    confs['extracao']['parametros']['tamanho'] = int(options.tamanho)
                    confs['extracao']['parametros']['qtde_patches'] = int(options.qtde_patches)
                    confs['extracao']['parametros']['sobrepoe'] = int(options.perc_sobrepoe)
                    
                except ValueError as e:
                    parser.error("Erro na conversao do valores de tamanho ou quantidade de patches")
        else:  # passado um tipo de extracao desconhecido
            parser.error("Tipo de extracao desconhecido.")
    else: # tipo de extracao não foi passado           
        parser.error("Tipo de extracao não informado.")
  
    
    logging.info("<<<<<<<< EXECUTA EXTRAÇÃO >>>>>>>>")
    t1 = time()
    
    for chave, item in confs.items():
        logging.info("{0} : {1}".format(str(chave), str(item)))
    
    patchExtrator = Extrator(confs)
    patchExtrator.executa()
            
    logging.info("\nTempo total da extracao: " + str(round(time()-t1,3)))
    
    # Encerramento e contagem de tempo de execucao 
    logging.info("Tempo total do programa: " + str(round(time()-t0,3)))
    logging.info("ENCERRAMENTO DO PROGRAMA \n\n")         
            
    
# Chamada programa principal  
if __name__ == "__main__":	
	main()
