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
from sklearn.metrics import f1_score
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

#verbose=-1,silent=True,
min_sum_hessian_in_leaf = 8
'''
#1.先训练分类问题，
cla=lgb.LGBMClassifier(objective='binary',num_leaves=5,learning_rate=0.01, n_estimators=720,
                              max_bin = 55, verbose=-1,silent=True, min_data_in_leaf =1
                              ,is_unbalance =True)

cla.fit(train_x,train_y>0)

#2.然后预测test数据集的类别
y_test_pred=cla.predict_proba(test_x)
thereshold=0.651
y_test_pred=y_test_pred[:,1]>thereshold#test


y_train_pred=cla.predict_proba(train_x)
y_train_pred=y_train_pred[:,1]>thereshold#test
print('precession:',y_train_pred[train_y>0].mean())
print('f1_score',f1_score(train_y>0,y_train_pred))


#3.训练回归模型，对train_y大于0的样本做回归模型的训练
gbm=lgb.LGBMRegressor( objective='regression',num_leaves=5,
                              learning_rate=0.01, n_estimators=80,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =1,verbose=-1,silent=True )

index_y=train_y>0
gbm.fit(train_x.iloc[index_y],np.log(train_y[index_y]),verbose=-1)


#4.把负类置零，正类用预测的结果。
predict_res=np.zeros(testID.shape)
predict_res[y_test_pred]= np.exp(gbm.predict(test_x.iloc[y_test_pred]))

#--save model ----------
#gbm.booster_.save_model('model.txt')
#print('save done')

print('fenlei feature importance:',cla.feature_importances_)
print('feature importance:',gbm.feature_importances_)

# output result
result = pd.DataFrame(testID,columns=['Id'])
result['Pred'] = predict_res

result.to_csv('./model/result.csv',header=True,index=False)

print("********FIT********:{:.2f} minute".format((datetime.datetime.now()-start_all).seconds/60))