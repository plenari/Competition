# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:59:18 2018

@author: omf
"""
import jieba 
import re as gensim
import numpy as np
import jieba.analyse
import pickle
import os
import config as c
import pandas as pd

class pre_train():
    '''
    1. 预处理好词向量模型和剪切好的单词。
    如果model和cut的数据都在，那就直接读。
    '''
    def __init__(self,):
        self.word2vec_path=c.word2vec_file#词向量模型
        self.word_cut_path=c.word_cut
    def process(self):
        '''处理流程'''
        if not os.path.exists(self.word_cut_path):
            '''如果没有cut单词，保证有单词'''
            self.read();self.cut();self.dump()
            
        if not os.path.exists(self.word2vec_path):
            '''这个词向量还没有测试'''
            data=self.load_word()
            q1,q2=data['q1'],data['q2']
            q1.extend(q2)
            self.word2vec(q1)
            
    def read(self,file=r'../data/nlp_train.csv'):
        '''
        读取数据。返回四个列表。分别是行号，问题，和标签
        '''  
        train=pd.read_csv(file,sep='\t',encoding='utf-8',header=None,index_col=None)
        
        self.index=list(np.arange(train.shape[0]))
        self.q1=train.loc[:,1]
        self.q2=train.loc[:,2]
        self.target=train.loc[:,3]
        
    def jieba_cut(self,sent):
        '''cut一句话        '''
        return list(jieba.cut(sent))
    
    def cut_sents(self,sents):
        '''cut 一个列表'''
        sents_cut=[]
        for i in sents:
            sents_cut.append(self.jieba_cut(i))
        return sents_cut
    
    def cut(self):
        '''cit 两个问题列表'''
        self.q1_cut=self.cut_sents(self.q1)
        self.q2_cut=self.cut_sents(self.q2)
        
    def dump(self,):
        '''把切割好的词汇保存到dump里'''
        obj={'index':self.index,'q1':self.q1_cut,'q2':self.q2_cut,'target':self.target}
        pickle.dump(obj,open(self.word_cut_path,'wb'))
                
    def load_word(self,):
        '''  load 一下切割好的数据  '''
        data=pickle.load(open(self.word_cut_path,'rb'))
        return data
    
    def word2vec(self,senten):
        model=gensim.models.Word2Vec(senten,min_count=1,iter=500)
        model.save('../data/word2vec_model')
       
    def doc2vec(self):
        pass
        
        
#test=pre_train()
#test.process()



#hb='为何我无法申请开通花呗信用卡收款'

def extra_tags_by_text(text,*arg):
    '''
    提取关键字并按照文章顺序返回。
    不能使用带权重和标签的返回值。 
    '''
    extra=jieba.analyse.extract_tags(text,*arg)
    index=[text.find(i) for i in extra ]
    index=np.argsort(index)
    extra=[extra[i] for i in index]
    return extra
