# coding=utf-8
'''
1. 这个准备用来写一个可以持续持续更新的模型
持续更新的模型是用dataset来喂数据的，

2.这些都是已经处理好的数据。

'''
import numpy as np
import pandas as pd
import lightgbm as lgb
import time

#-------rrrrrrrrrrrrrrread data-------
train_x=pd.read_csv('data2/train_x.csv',index_col=0)
test_x=pd.read_csv('data2/test_x.csv',index_col=0)
test_x=test_x[train_x.columns]
train_y=pd.read_csv('data2/train_y.csv',index_col=0,header=None)
res=pd.read_csv('data2/test1.csv')


#----------------Dataset-----------------
# if you want to re-use data, remember to set free_raw_data=False
lgb_train = lgb.Dataset(train_x, train_y,\
            free_raw_data=False)

#--------------set paras-------------
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 20,
    'learning_rate': 0.10,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 10,
    'verbose': 0,
    'reg_lambda':1,
    'early_stopping_rounds':100,  
    'subsample':0.8,
    'subsample_freq':30, 
    'min_data_in_leaf':1,
}

model_name=None
#----------------model --------------
estimator = lgb.train(params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=lgb_train,
                init_model=model_name,
                keep_training_booster=True
                )

#------------------explain---------------------
'''
subsample_for_bin 构建直方图的样本数量
min_split_gain    在叶节点进一步区分需要的最小增益
min_child_weight  字节点的最小权重
min_child_samples 子节点最少的样本数
subsample         训练集的采样比率
subsample_freq    子采样的频率？？
colsample_bytree  构建每棵树，特征的采样比率
'''

#----------------save model---------------------------
now=time.strftime('%m-%d-%H-%M',time.localtime())
try:
    estimator.save_model('model2-{}.txt'.format(now))
except Exception as e:
    print('errors:',e)

#--------------------prediction-----------------------
res['score'] = estimator.predict(test_x)
res['score'] = res['score'].apply(lambda x: float('%.8f' % x))
res.to_csv('data2/submission-{}.csv'.format(now), index=False)

