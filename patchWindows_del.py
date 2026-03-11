import pandas as pd
import numpy as np
import pyBigWig as pbw
import sys
import os
import gzip
from copy import deepcopy
import glob

path = sys.argv[1]
species = sys.argv[2]
annot = sys.argv[3]
chrm = sys.argv[4]
delpath = sys.argv[5]

bw = pbw.open("root_"+species+"/"+annot+"/genomic_features/genedensity.bw")
n = bw.chroms(chrm)

os.makedirs("./delfiles", exist_ok=True)

del_df = pd.read_csv(delpath,header=None,sep='\t')
del_df = del_df[del_df[0] == chrm]

for di in range(del_df.shape[0]):
    width = 524288
    startM = del_df.iloc[di,1] - 262144
    startU = del_df.iloc[di,1]
    startD = del_df.iloc[di,1] - 524288
    if startM < 0 or startU < 0 or startD < 0:
        continue
    ds_M = del_df.iloc[di,1] - startM
    de_M = ds_M + (del_df.iloc[di,2] - del_df.iloc[di,1] + 1)

    pathM = path+'/npy/'+species+'/'+annot+'/'+del_df.iloc[di,0]+'_'+str(startM)+'_'+del_df.iloc[di,3].lower()+'.npy'
    pathU = path+'up'+'/npy/'+species+'/'+annot+'/'+del_df.iloc[di,0]+'_'+str(startU)+'_'+del_df.iloc[di,3].lower()+'.npy'
    pathD = path+'down'+'/npy/'+species+'/'+annot+'/'+del_df.iloc[di,0]+'_'+str(startD)+'_'+del_df.iloc[di,3].lower()+'.npy'
    if (not os.path.isfile(pathM)) or (not os.path.isfile(pathU)) or (not os.path.isfile(pathD)):
        continue
    
    indvempty_n = np.zeros(n,dtype='float')
    mask_n = np.zeros(n,dtype='int32')

    fileM = np.load(pathM)
    indvempty_nM = np.zeros(width,dtype='float32')
    indvempty_nM[:100000] += np.mean(fileM[:100000])
    indvempty_nM[100000:200000] += np.mean(fileM[100000:200000])
    indvempty_nM[200000:300000] += np.mean(fileM[200000:300000])
    indvempty_nM[300000:400000] += np.mean(fileM[300000:400000])
    indvempty_nM[400000:500000] += np.mean(fileM[400000:500000])
    indvempty_nM[500000:width] += np.mean(fileM[500000:width])

    fileU = np.load(pathU)
    indvempty_nU = np.zeros(width,dtype='float32')
    indvempty_nU[:100000] += np.mean(fileU[:100000])
    indvempty_nU[100000:200000] += np.mean(fileU[100000:200000])
    indvempty_nU[200000:300000] += np.mean(fileU[200000:300000])
    indvempty_nU[300000:400000] += np.mean(fileU[300000:400000])
    indvempty_nU[400000:500000] += np.mean(fileU[400000:500000])
    indvempty_nU[500000:width] += np.mean(fileU[500000:width])

    fileD = np.load(pathD)
    indvempty_nD = np.zeros(width,dtype='float32')
    indvempty_nD[:100000] += np.mean(fileD[:100000])
    indvempty_nD[100000:200000] += np.mean(fileD[100000:200000])
    indvempty_nD[200000:300000] += np.mean(fileD[200000:300000])
    indvempty_nD[300000:400000] += np.mean(fileD[300000:400000])
    indvempty_nD[400000:500000] += np.mean(fileD[400000:500000])
    indvempty_nD[500000:width] += np.mean(fileD[500000:width])

    indvempty_n[startM:startM+ds_M] += indvempty_nM[:ds_M]
    mask_n[startM:startM+ds_M] += 1
    endM_len = indvempty_nM[de_M:].shape[0]
    indvempty_n[startM+de_M:startM+de_M+endM_len] += indvempty_nM[de_M:]
    mask_n[startM+de_M:startM+de_M+endM_len] += 1

    endU_len = indvempty_nU[(del_df.iloc[di,2]-del_df.iloc[di,1]+1):].shape[0]
    indvempty_n[startU+(del_df.iloc[di,2]-del_df.iloc[di,1]+1):startU+(del_df.iloc[di,2]-del_df.iloc[di,1]+1)+endU_len] += indvempty_nU[(del_df.iloc[di,2]-del_df.iloc[di,1]+1):]
    mask_n[startU+(del_df.iloc[di,2]-del_df.iloc[di,1]+1):startU+(del_df.iloc[di,2]-del_df.iloc[di,1]+1)+endU_len] += 1

    indvempty_n[startD:startD+width] += indvempty_nD
    mask_n[startD:startD+width] += 1

    indvempty_n = np.divide(indvempty_n,mask_n)
    np.save('delfiles/'+del_df.iloc[di,0]+'_predicted_'+str(startM)+'_'+del_df.iloc[di,3].lower()+'.npy',indvempty_n)
