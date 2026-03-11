import pandas as pd
import numpy as np
import pyBigWig as pbw
import sys
import os
from copy import deepcopy
import glob

path = sys.argv[1]
species = sys.argv[2]
annot = sys.argv[3]
chrm_list = sys.argv[4].split(',')

os.makedirs("./bedfiles", exist_ok=True)

bw = pbw.open("root_"+species+"/"+annot+"/genomic_features/genedensity.bw")
for chrm in chrm_list:
    chrm = chrm.split("chr")[-1]
    print("chr"+str(chrm))
    n = bw.chroms("chr"+str(chrm))
    empty_n = np.load(path+'/npy/'+species+'/'+annot+'/chr'+str(chrm)+'_predicted.npy')
    empty_n = np.concatenate((empty_n,[0]))
    empty_n[np.isnan(empty_n)] = 0
    cut = 0.2
    empty_n = np.array(empty_n >= cut, dtype='float')
    empty_n[empty_n < cut] = 0
    pos_x = []
    flag = False
    for i in range(empty_n.shape[0]):
        if empty_n[i]:
            if not flag:
                flag = True
                pos_x.append(i)
        else:
            if flag:
                flag = False
                pos_x.append(i-1)
    vals = []
    for i in range(int(len(pos_x)/2)):
        vals.append(['chr'+str(chrm),pos_x[2*i],pos_x[(2*i)+1]+1])

    df = pd.DataFrame(vals)
    df.to_csv('bedfiles/chr'+str(chrm)+'_hg19_predicted_lad.bed',header=None,sep='\t',index=None)


