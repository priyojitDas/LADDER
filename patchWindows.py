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

bw = pbw.open("root_"+species+"/"+annot+"/genomic_features/genedensity.bw")
for chrm in chrm_list:
    chrm = chrm.split("chr")[-1]
    n = bw.chroms("chr"+str(chrm))
    bw_val = np.array(bw.values('chr'+str(chrm),0,n))
    bw_val = np.array(bw_val >= 0.75, dtype='float')
    empty_n = np.zeros(n,dtype='float32')
    mask_n = np.zeros(n,dtype='int32')
    width = 524288 
    nfiles = len(glob.glob(path+'/npy/'+species+'/'+annot+'/chr'+str(chrm)+'_*.npy'))
    print("# of windows:",'chr'+str(chrm),nfiles)
    for ipos in range(nfiles):
        istart = ipos * 200000
        if os.path.exists(path+'/npy/'+species+'/'+annot+'/chr'+str(chrm)+'_'+str(istart)+'.npy'):
            file = np.load(path+'/npy/'+species+'/'+annot+'/chr'+str(chrm)+'_'+str(istart)+'.npy')
        else:
            continue
        empty_n[istart:istart+100000] += np.mean(file[:100000])
        empty_n[istart+100000:istart+200000] += np.mean(file[100000:200000])
        empty_n[istart+200000:istart+300000] += np.mean(file[200000:300000])
        empty_n[istart+300000:istart+400000] += np.mean(file[300000:400000])
        empty_n[istart+400000:istart+500000] += np.mean(file[400000:500000])
        empty_n[istart+500000:istart+width] += np.mean(file[500000:width])
        mask_n[istart:istart+width] += 1 
    empty_n = np.divide(empty_n, mask_n)
    np.save(path+'/npy/'+species+'/'+annot+'/chr'+str(chrm)+'_predicted.npy',empty_n)
