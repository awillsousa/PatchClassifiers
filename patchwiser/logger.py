# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 12:41:03 2016
@author: antoniosousa
"""

import logging
import datetime

# Define o formato da data e hora a ser incluido no nome do arquivo de log
hoje = datetime.datetime.today()
formato = "%a-%b-%d-%H_%M_%S"
arq_log = "./"+hoje.strftime(formato)+".log"
# Configura logging para arquivo 

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=arq_log,
                    filemode='w')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
log = logging.getLogger()
log.addHandler(console)
