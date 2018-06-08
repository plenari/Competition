#coding utf-8
import pandas as pd
import numpy as np  
import lightgbm as lgb
import datetime
from prepro import deal_train,deal_test
from sklearn import model_selection

def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
'''
交叉验证:
    * 利用交叉验证预测出来test的值，然后去平均
    * 保存交叉验证在训练集上的结果，并保存下来计算gini

'''

start_all0 = datetime.datetime.now()

test_x,testID=deal_test()
train_x,train_y=deal_train()
train_x.fillna(0)
test_x.fillna(0)
print("*******DATA*******:{:.2f} hours".format((datetime.datetime.now()-start_all0).seconds/3600))
start_all = datetime.datetime.now()

#-----model------
n_split=5
cv_split = model_selection.KFold(n_splits=n_split, random_state=15, shuffle=False)
gbm=lgb.LGBMRegressor( objective='regression',num_leaves=5,
                              learning_rate=0.01, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =1, min_sum_hessian_in_leaf = 11,min_data=1 )


#保存每个交叉验证在y_test集上的结果
y_pred_test_cv=np.ones((test_x.shape[0],n_split))
y_pred_train_cv=np.ones((train_x.shape[0],))

for i, (train_index,test_index) in enumerate(cv_split.split(train_x,train_y)):        
        
    gbm.fit(train_x.iloc[train_index],train_y[train_index],eval_set=[(train_x.iloc[test_index],train_y[test_index])],
            early_stopping_rounds=50,verbose=False,eval_metric='l1')
    
    #print gini 
    y_test=train_y[test_index]  
    y_predict=gbm.predict(train_x.iloc[test_index])
    print('*'*40)
    print('cv:',i,'gini:',gini(y_test,y_predict))      
    print('*'*10,'FEATURE_IMPORTSNCES_','*'*10)
    print(gbm.feature_importances_)

        
    #save y_pred_i
    y_pred_test_cv_i=gbm.predict(test_x)
    y_pred_test_cv[:,i]=y_pred_test_cv_i
        

#
print('\nGini in train_x,train_y:',gini(train_y,y_pred_test_cv.mean(axis=1)))
#cal y_pred
y_pred=y_pred_test_cv.mean(axis=1)

#把小于0的改成等于0
#y_pred[y_pred<0]=0

# output result
result = pd.DataFrame(testID,columns=['Id'])
result['Pred'] = y_pred

result.to_csv('./model/result.csv',header=True,index=False)

print("********FIT********:{:.2f} hours".format((datetime.datetime.now()-start_all).seconds/3600))
