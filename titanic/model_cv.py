# -*- coding: utf-8 -*-
"""
Created on Sat May  5 15:09:07 2018
@author:Plenari
@email:shengjiex@qq.com
"""
import pandas as pd
import numpy as np  
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest 
from sklearn.model_selection import ParameterGrid, cross_val_score, StratifiedKFold, learning_curve
from sklearn import preprocessing
import gc
from sklearn.metrics import accuracy_score
import maplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier)


train_x = pd.read_csv('data2/train_x.csv',index_col=0)
test_x = pd.read_csv('data2/test_x.csv',index_col=0)
res_label=np.load('data2/res_label.npz')
res,train_y=res_label['res'],res_label['label']
print(train_x.shape,test_x.shape,train_y.shape,res.shape)

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
            i['score']=accuracy_score(train_y,clf.predict(train_x))
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
lgb_paras={'boosting_type':['gbdt','dart','rf'],
           'num_leaves':[20,30,40],
           'n_estimators':[100,500],
           'min_child_samples':[5,10,20,],
           'subsample':[0.5,0.8]    ,
           'subsample_for_bin':[1000],          
}

#grid search 
lr_res=ParameterGrid_run(LogisticRegression,lr_paras,train_x,train_y)
lgb_res=ParameterGrid_run(lgb.LGBMClassifier,lgb_paras,train_x,train_y)

#find the best paras
lr_para=lr_res.iloc[0].to_dict()
lgb_para=lgb_res.iloc[0].to_dict()
del lgb_para['score'],lr_para['score']
gc.collect()
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



