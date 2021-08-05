#!/bin/bash

ARQLOG=$1

grep -v -E '(Classe predi|Tempo classifica|Classificacao imagem|Contagem |Extracao imagem|Tempo extra)' $ARQLOG | sed 's/^.*log\://g'


