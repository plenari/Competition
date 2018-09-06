# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:59:18 2018
@author: omf
"""
import jieba 
import pandas as pd
import numpy as np
import jieba.analyse
import gensim
from gensim.models import word2vec
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn import decomposition
#myself
from preprocess import pre_train
import config as c

load_word=pre_train()

class datasets(Dataset):
    '''批量训练
    '''
    def __init__(self):        
        self.word2vec_model=c.word2vec_file
        self.word_cut=c.word_cut
        
        self.data=load_word.load_word()
        self.word2vec_model=gensim.models.KeyedVectors.load_word2vec_format(self.word2vec_model)
    def __len__(self):
        #return self.lens
        return len(self.data['index'])
    
    def __getitem__(self,idx):
        lines=self.data['index'][idx]
        q1=self.data['q1'][idx]
        q2=self.data['q2'][idx]
        target=self.data['target'][idx]   
        
        q1=self.word2vec_q(q1)# 把他们变成n*2
        q2=self.word2vec_q(q2)
        
        return lines,q1,q2,target
    
    def cut(self,sentences):
        ''' jieba cut'''
        return list(jieba.cut(sentences))
        
    def word2vec_q(self,sentences):
        '''        
        把每个问题对应的句子转变成vec
        '''
        re=[]
        for i in sentences:
            try:#不知道在不在词汇里。
                re.append(self.word2vec_model[i])
            except:
                pass
        res=self.mean(np.array(re))
        return res
    
    def mean(self,array):
        '''
        先求平均吧,
        '''
        return torch.tensor(np.mean(array,axis=0).reshape(1,-1))
    
    def LDA(self,array,dim=2):
        '''
        lda  好像有点问题。
        '''
        lda=decomposition.LatentDirichletAllocation(n_components=dim)
        return lda.fit_transform(array)
    
    
#t=datasets()
#f=DataLoader(t,batch_size=50)
