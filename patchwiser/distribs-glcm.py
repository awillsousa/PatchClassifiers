#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 17:51:37 2017

@author: willian

Exibe as distribuições de uma base GLCM. Para cada atributo é apresentada uma distribuição

"""

from optparse import OptionParser
import helper as hp
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

cores = ['blue', 'green', 'red', 'grey',  'purple',
         'magenta', 'yellow', 'cyan',
         'indigo', 'tomato', 'maroon', 'gold', 
         'crimson', 'teal', 'firebrick'] 

atribs_glcm = """Energy,Contrast,Correlation,Variance,
Inverse Different Moment,Sum Average,Sum Variance,
Sum Entropy,Entropy,Difference Variance,Difference Entropy,
Information Measure Correlation 1,Information Measure Correlation 2,
Maximal Correlation Coeficient""".replace('\n','').split(',')

atribs_exibe = [0,1,2,4,5,8]

def histos_glcm_fold(folds, titulo):
        
    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, sharey=True)
        
    for fold in folds: 
        ncols = fold.shape[1]
        for i in atribs_exibe:            
            col = fold[:,i]        
            if i == 0:
                col = np.sqrt(col)
               
                
            ax1.hist(np.asarray(col), bins='auto', histtype='stepfilled', normed=True, color=cores[i], label=atribs_glcm[i])
            ax1.set_title(titulo)
            ax2.hist(np.asarray(col), bins='auto', histtype='stepfilled', normed=True, color=cores[i], label=atribs_glcm[i])
            ax3.hist(np.asarray(col), bins='auto', histtype='stepfilled', normed=True, color=cores[i], label=atribs_glcm[i])
            ax4.hist(np.asarray(col), bins='auto', histtype='stepfilled', normed=True, color=cores[i], label=atribs_glcm[i])
            ax5.hist(np.asarray(col), bins='auto', histtype='stepfilled', normed=True, color=cores[i], label=atribs_glcm[i])
            # Fine-tune figure; make subplots close to each other and hide x ticks for
            # all but bottom plot.
            f.subplots_adjust(hspace=0)
            plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)



def histogramas_glcm(base, titulo):    
    ncols = base.shape[1]
    
    plt.xlabel('')
    plt.ylabel('Patches')
    plt.title(titulo)
    plt.ylim([0.0, 50])
    plt.xlim([0.0, 1.1])
    #for i in range(0, ncols):
    for i in atribs_exibe:            
        col = base[:,i]        
        if i == 0:
            col = np.sqrt(col)
            
        plt.hist(np.asarray(col), bins='auto', histtype='stepfilled', normed=True, color=cores[i], label=atribs_glcm[i])
        arquivo = "histo-base_glcm.pdf"        
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=3)
    plt.savefig(arquivo)
    #plt.clf()
    #plt.show()    

def histogramas_mag(bases, titulo):    
    has_leg = False
    
    plt.xlabel('')
    plt.ylabel('Num. Patches')
    plt.title(titulo)
    plt.ylim([0.0, 50])
    plt.xlim([0.0, 1.1])
    #for i in range(0, ncols):
    for base in bases:
        for i in atribs_exibe:            
            col = base[:,i]        
            if i == 0:
                col = np.sqrt(col)
            
            if not has_leg:
                plt.hist(np.asarray(col), bins='auto', histtype='stepfilled', normed=True, color=cores[i], label=atribs_glcm[i])
            else:
                plt.hist(np.asarray(col), bins='auto', histtype='stepfilled', normed=True, color=cores[i])
                    
        has_leg = True
        
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=3)
    arquivo = "histo-"+titulo+".png"
    plt.savefig(arquivo)
    plt.show()    
    plt.clf()
    
# PROGRAMA PRINCIPAL 
def main():
    
    folds = ['fold1','fold2','fold3','fold4','fold5']
    mags = ['40X', '100X', '200X', '400X']    
    base="/home/willian/bases/execs/cancer/"    
    bases_mags = {m:[] for m in mags}
    
    for mag in mags:            
        for i,fold in enumerate(folds):    
            base_glcm = base+fold+"/train/"+mag+"/base_pftas_ptx64x64.glcm"    
            titulo = " - ".join([fold,mag])
            BASE_GLCM, _,_ = hp.carrega_base(base_glcm,n_features=14)           
            BASE_GLCM = BASE_GLCM.toarray()
            BASE_GLCM = MinMaxScaler().fit_transform(BASE_GLCM)
            
            bases_mags[mag].append(BASE_GLCM) 
            #histogramas_glcm(BASE_GLCM, titulo)
        histogramas_mag(bases_mags[mag], " Atrib. GLCM {0}".format(mag))
            
# Chamada programa principal  
if __name__ == "__main__":	
	main()