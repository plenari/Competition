# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 20:57:38 2018

@author: omf
"""
#     1.1 模型处理
#         1.2 获取数据
#         1.3 模型
#         1.4 多个模型

import jieba 
import gensim

from sklearn.metrics import f1_score
from preprocess import pre_train
import config as c
import pandas as pd
import numpy as np  
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest 
from sklearn.model_selection import ParameterGrid, cross_val_score, StratifiedKFold, learning_curve
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier)


#word2vec_model=gensim.models.KeyedVectors.load_word2vec_format(c.word2vec_file)
#load_word=pre_train()
#data=load_word.load_word()
# data_feed
def get_data_mean(idx):
    '''
    就是把数据传入然后求平均
    '''
    q1=np.zeros((len(idx),c.word_dim))
    q2=np.zeros((len(idx),c.word_dim))    
    for i,j in enumerate(idx):        
        q1[i]=transform_word_vector_mean(data['q1'][j])
        q2[i]=transform_word_vector_mean(data['q2'][j])        
    target=np.array([data['target'][i] for i in idx])
    return q1,q2,target 

def transform_word_vector_mean(word_lists):
    '''    分好词的句子的列表'''
    re=np.zeros((len(word_lists),c.word_dim))
    for i,word in enumerate(word_lists):
        if word  in word2vec_model.vocab:
            re[i]=word2vec_model[word]
    return re.mean(axis=0)

q1,q2,train_y=get_data_mean(np.arange(len(data['index'])))
train_x=np.hstack([q1,q2])

#-----------------Gridsearch ---------
def ParameterGrid_run(func,paras,train_x,train_y):
    '''
    func:estimator
    paras:参数
    train_x:
    train_y:
    '''
    PG=ParameterGrid(paras)
    res=[]
    for i in PG:
        try:
            clf=func(**i)
            clf.fit(train_x,train_y)
            i['score']=f1_score(train_y,clf.predict(train_x))
            res.append(i)
        except Exception as e:
            print(e)
    #还缺一个保存结果的文件。比如保存得分前十的结果和模型
    
    res=pd.DataFrame(res)
    return res.sort_values('score',ascending=False).reset_index(drop=True)


def plot_learing_curve(clf,train_x,train_y):
    '''
    clf :estimator .
    
    '''
    a=learning_curve(clf,train_x,train_y,verbose=1)
    plt.plot(a[0],a[1].mean(axis=1),label='train')
    plt.plot(a[0],a[2].mean(axis=1),label='eval')
    plt.legend()
    plt.show()

#-------------params----------------  
lr_paras={"max_iter":[100,500],
          'C':[0.5,0.8,1.0],
          'verbose':[1],
          'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}
 
lgb_paras={'boosting_type':['gbdt'],
           'num_leaves':[5,10,15],
           'n_estimators':[50,100],
           'min_child_samples':[5,10,20,],
           'subsample':[0.5,0.8]    ,
           'subsample_for_bin':[100],          
           'bagging_freq': [3,5],
           'objective': ['binary'],
           'metric': ['auc'], 
           'njobs':[-1],
           }


#grid search 
lr_res=ParameterGrid_run(LogisticRegression,lr_paras,train_x,train_y)
lgb_res=ParameterGrid_run(lgb.LGBMClassifier,lgb_paras,train_x,train_y)

#find the best paras
lr_para=lr_res.iloc[0].to_dict()
lgb_para=lgb_res.iloc[0].to_dict()
del lgb_para['score'],lr_para['score']

#train and save:
    
clf=lgb.LGBMClassifier(**lgb_para)
clf.fit(train_x,train_y)

clf=LogisticRegression(**lr_para)
clf.fit(train_x,train_y)
#--blending----
def get_oof(clf, train_x, train_y, test_x):
    '''
    blednding
    
    clf:estimator
    train_x: pd.DataFrame
    train_y: np.array
    test_x: pd.DataFrame
    '''
    NFolds=5
    len_train_x=train_x.shape[0]
    len_test_x=test_x.shape[0]
    oof_train = np.zeros((len_train_x,))
    oof_test = np.zeros((len_test_x,))
    oof_test_skf = np.empty((NFolds, len_test_x))#5-170
    kf=StratifiedKFold(n_splits=NFolds)
    for i, (train_index, test_index) in enumerate(kf.split(train_x,train_y)):
        x_tr = train_x.iloc[train_index]
        y_tr = train_y[train_index]
        x_te = train_x.iloc[test_index]

        clf.fit(x_tr, y_tr)
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(test_x)

    oof_test = oof_test_skf.mean(axis=0)
    #return oof_train, oof_test
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

'''
训练没有test数据，所以先不算的。
'''


#y_pred=[int(i>0.16) for i in y]
#print('f1_score',i,f1_score(train_y,y_pred))

# * 保存模型
#estimator.save_model('../model/lightgbm_model')
# * load 模型
#bst = lightgbm.Booster(model_file='../model/lightgbm_model')
