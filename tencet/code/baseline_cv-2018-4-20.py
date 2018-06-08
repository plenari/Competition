# coding=utf-8
# @author:bryan
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import pickle
import time
#
ad_feature=pd.read_csv('data/adFeature.csv')
user_feature=pd.read_csv('data/userFeature.csv')
train=pd.read_csv('data/train.csv')
predict=pd.read_csv('data/test1.csv')
#
train.loc[train['label']==-1,'label']=0
predict['label']=-1
#
data=pd.concat([train,predict])
data=pd.merge(data,ad_feature,on='aid',how='left')
data=pd.merge(data,user_feature,on='uid',how='left')
data=data.fillna('-1')
#
one_hot_feature=['LBS','age','carrier','consumptionAbility','education',\
                 'gender','house','os','ct','marriageStatus','advertiserId',\
                 'campaignId', 'creativeId', 'adCategoryId', 'productId', 'productType']

vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3',\
                'interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']

for feature in one_hot_feature:
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])

train=data[data.label!=-1]
train_y=train.pop('label')

# train, test, train_y, test_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
test=data[data.label==-1]
res=test[['aid','uid']]
test=test.drop('label',axis=1)
enc = OneHotEncoder()
enc.fit(data['creativeSize'].values.reshape(-1,1))
train_x=enc.transform(train['creativeSize'].values.reshape(-1,1))
test_x=enc.transform(test['creativeSize'].values.reshape(-1,1))


#
for feature in one_hot_feature:
    enc.fit(data[feature].values.reshape(-1, 1))
    train_a=enc.transform(train[feature].values.reshape(-1, 1))
    test_a = enc.transform(test[feature].values.reshape(-1, 1))
    train_x= sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
print('one-hot prepared !')

cv=CountVectorizer()
for feature in vector_feature:
    cv.fit(data[feature])
    train_a = cv.transform(train[feature])
    test_a = cv.transform(test[feature])
    train_x = sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
print('cv prepared !')

#------------------模型---------------------

estimator = lgb.LGBMClassifier(
        boosting_type='gbdt', reg_alpha=0.0,max_depth=-1, n_estimators=3000,\
        subsample_for_bin=50000,objective='binary',colsample_bytree=0.7, \
        subsample_freq=1,learning_rate=0.05,min_child_weight=0.0001, n_jobs=10,
    )
'''
subsample_for_bin 构建直方图的样本数量
min_split_gain    在叶节点进一步区分需要的最小增益
min_child_weight  字节点的最小权重
min_child_samples 子节点最少的样本数
subsample         训练集的采样比率
subsample_freq    子采样的频率？？
colsample_bytree  构建每棵树，特征的采样比率
'''
#GridSearchCV(estimator, param_grid, scoring=None, \
#fit_params=None, n_jobs=1, iid=True, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score='raise', return_train_score='warn')
param_grid = {
        'reg_lambda':[0.5,0.9],#L2正则
         'num_leaves':[31,50],  #叶子节点
         'min_split_gain':[0.,0.03]#最小的增益
}
fit_param={
    'eval_metric':'auc',
    'early_stopping_rounds':100,
    'eval_set':[(train_x,train_y)]
}
gdcv = GridSearchCV(estimator, param_grid,fit_params=fit_param,n_jobs=1)
gdcv.fit(train_x, train_y)

#----------------save model---------------------------
now=time.strftime('%m-%d-%H-%M',time.localtime())
with open('GridSearchCV-{}.pkl'.format(now),'wb') as f:
    '''
    保存交叉验证的数据，load不能直接使用
    '''
    pickle.dump(gdcv,f)

with open('best-estimator-{}.pk'.format(now),'wb') as f:
    '''
    保存最好的模型，load后可以直接用来预测或者继续训练
    '''
    pickle.dump(gdcv.best_estimator_.booster_,f)
#-------------------------------------------
#预测
res['score'] = gdcv.best_estimator_.predict_proba(test_x)[:,1]
res['score'] = res['score'].apply(lambda x: float('%.8f' % x))
res.to_csv('data/submission-{}.csv'.format(now), index=False)

