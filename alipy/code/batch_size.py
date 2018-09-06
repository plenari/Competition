# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:59:18 2018
@author: omf
"""
from torch.utils.data import Dataset, DataLoader
from preprocess import pre_train
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

class datasets(Dataset):
    
    '''batch_size  返回行号索引。
    只返回行号索引怎么样？
    '''
    def __init__(self,sample=False):       
        load_word=pre_train()        
        data=load_word.load_word()
        if sample:        
            rus = RandomUnderSampler()
            index,_ = rus.fit_sample(np.array(data['index']).reshape(-1,1), data['target'])
            index=index.flatten()
        else:
            index=list(np.arange(len(data['index'])))
        self.data=index    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        lines=self.data[idx]
        return lines
    
    
#t=datasets(True)
#f=DataLoader(t,batch_size=50)
