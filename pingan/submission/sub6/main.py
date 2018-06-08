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

import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
start_all0 = datetime.datetime.now()
#test_x,testID=deal_test()
#train_x,train_y=deal_train()
train_x.fillna(0)
test_x.fillna(0)

print("*******DATA*******:{:.2f} minute".format((datetime.datetime.now()-start_all0).seconds/60))

start_all = datetime.datetime.now()

gbm=lgb.LGBMRegressor( objective='regression',num_leaves=5,
                              learning_rate=0.01, n_estimators=500,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =1, min_sum_hessian_in_leaf = 11,min_data=1 )
    
gbm.fit(train_x,train_y)



print('feature importance:',gbm.feature_importances_)
#--save model ----------
#gbm.booster_.save_model('model.txt')
#print('save done')
# output result
result = pd.DataFrame(testID,columns=['Id'])
result['Pred'] = gbm.predict(test_x)

result.to_csv('./model/result.csv',header=True,index=False)

print("********FIT********:{:.2f} minute".format((datetime.datetime.now()-start_all).seconds/60))