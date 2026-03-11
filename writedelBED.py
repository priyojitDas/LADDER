import pandas as pd
import numpy as np
import pyBigWig as pbw
import sys
import os
from copy import deepcopy
import glob

path = sys.argv[1]

file_list = glob.glob(path+'/*.npy')
for file in file_list:
    chrm = file.split('/')[-1].split('_')[0].split('chr')[1]
    empty_n = np.load(file)
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
    df.to_csv('delfiles/'+file.split('/')[-1].split('.')[0]+'.bed',header=None,sep='\t',index=None)


