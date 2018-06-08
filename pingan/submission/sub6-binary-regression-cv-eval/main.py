# -*- coding: utf-8 -*-
"""
Created on Thu May  3 16:12:25 2018
@email:shengjiex@qq.com
@author: Plenari
"""
import pandas as pd
import numpy as np  
import lightgbm as lgb
import datetime
from prepro import deal_train,deal_test
#from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold,KFold
#from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
start_all0 = datetime.datetime.now()
test_x,testID=deal_test()
train_x,train_y=deal_train()
train_x.fillna(0)
test_x.fillna(0)
print("*******DATA*******:{:.2f} minute".format((datetime.datetime.now()-start_all0).seconds/60))
start_all = datetime.datetime.now()

'''
1.先训练分类问题，
2.然后预测test数据集的类别
3.训练回归模型，对train_y大于0的样本做回归模型的训练
3.预测出来的正类在进行回归
4.把负类置零，正类用预测的结果。
'''
def LGB_class(train_x,train_y,n_split=5):
    '''   
    train_x array 
    train_y array
    train_y 需要是类别
    '''
    #1.先训练分类问题，   
    cv_split = StratifiedKFold(n_splits=n_split, random_state=None )
    
    #用来分类的 y
    best_score=0.5
    best_model=None
    for i, (train_index,test_index) in enumerate(cv_split.split(train_x,train_y)):     
        '''        
        交叉验证分类问题，保留最好的模型
        '''
        
        clf=lgb.LGBMClassifier(objective='binary',num_leaves=20,learning_rate=0.01, n_estimators=720,
                                  max_bin = 55, verbose=-1,silent=True, min_data_in_leaf =1
                                  ,is_unbalance =True)
        
        clf.fit(train_x[train_index],train_y[train_index],\
                eval_set=[(train_x[test_index],train_y[test_index])], \
                eval_metric='auc', verbose=-1, early_stopping_rounds=200)
        #use the best model
        score=clf.best_score_.get('valid_0')['auc']
        if score>best_score:
            best_model=clf
            best_score=score
            #保存训练集结果
            y_predict_train=clf.predict(train_x)
            
    return best_model,best_score,y_predict_train



def LGB_regre(train_x,train_y,n_split=5):
    '''   
    train_x array 
    train_y array
    
    '''
    #1.先训练分类问题，   
    cv_split = KFold(n_splits=n_split, random_state=None )
    
    #用来分类的 y
    best_score=np.inf
    best_model=None
    for i, (train_index,test_index) in enumerate(cv_split.split(train_x,train_y)):     
        '''        
        交叉验证回归问题，保留最好的模型
        '''
        
        clf=lgb.LGBMRegressor( objective='regression',num_leaves=5,
                                  learning_rate=0.01, n_estimators=100,
                                  max_bin = 55, bagging_fraction = 0.8,
                                  bagging_freq = 5, feature_fraction = 0.2319,
                                  feature_fraction_seed=9, bagging_seed=9,
                                  min_data_in_leaf =1,verbose=-1,silent=True )
    
        clf.fit(train_x[train_index],train_y[train_index],\
            eval_set=[(train_x[test_index],train_y[test_index])], \
            eval_metric='l1', verbose=-1, early_stopping_rounds=10)
        #use the best model
        score=clf.best_score_.get('valid_0')['l1']
        if score<best_score:
            best_model=clf
            best_score=score
            #保存训练集结果
            y_predict_train=clf.predict(train_x)
                    
    return best_model,best_score,y_predict_train,

#训练分类模型，
cla_model,cla_score,cla_train_y=LGB_class(train_x.values,train_y>0)
#预测测试集类别
cla_test_y=cla_model.predict(test_x)
#用train>0的数据，训练回归模型，并预测测试集正类的回归值

train_y_bt_0=train_y>0  

reg_model,reg_sroce,reg_train_y=LGB_regre(train_x.values[train_y_bt_0],train_y[train_y_bt_0])

#-----------看一下在训练集上的情况-------
y_pred_train=np.zeros(train_y.shape)
y_pred_train[cla_train_y]=reg_model.predict(train_x.values[cla_train_y])
print('corrcoef in train:',np.corrcoef(train_y,y_pred_train)[0,1])

#----------------------------------
print('score:',cla_score,reg_sroce)
print('binary:',cla_model.feature_importances_)
print('regression:',reg_model.feature_importances_)
#-------------------

y_test_=reg_model.predict(test_x.values[cla_test_y])
y_pred_test_res=np.zeros(testID.shape)
y_pred_test_res[cla_test_y]=y_test_
##################################################
#--save model ----------
#gbm.booster_.save_model('model.txt')
#print('save done')

# output result
result = pd.DataFrame(testID,columns=['Id'])
result['Pred'] = y_pred_test_res

result.to_csv('./model/result.csv',header=True,index=False)

print("********FIT********:{:.2f} minute".format((datetime.datetime.now()-start_all).seconds/60))