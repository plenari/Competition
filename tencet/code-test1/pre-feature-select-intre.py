# coding=utf-8
# @author:bryan
#@author:Plenari
'''
对文本数据特征选择:
'appIdAction','appIdInstall',
'interest1','interest2','interest3','interest4','interest5',
'kw1','kw2','kw3',
'topic1','topic2','topic3'
'''
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.feature_selection import SelectKBest,SelectPercentile,f_classif
from scipy import sparse
import os
from sklearn.externals import joblib
import numpy as np

#----------read---------
#ad_feature=pd.read_csv('data/adFeature.csv')
#user_feature=pd.read_csv('data/userFeature.csv')
#train=pd.read_csv('data/train.csv')
#predict=pd.read_csv('data/test1.csv')

#train.loc[train['label']==-1,'label']=0
#predict['label']=-1
#data=pd.concat([train,predict])
#data=pd.merge(data,ad_feature,on='aid',how='left')
#data=pd.merge(data,user_feature,on='uid',how='left')
#data=data.fillna('-1')
#data.to_csv('data_after_merge.csv',index=None)
#这个应该是含有aid和uid的
data=pd.read_csv('data_after_merge.csv',index_col=None)

#-----------feature--------------
one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house',\
                 'os','ct','marriageStatus','advertiserId','campaignId', 'creativeId',
                'adCategoryId', 'productId', 'productType']
vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4',\
                'interest5','kw1','kw2','kw3','topic1','topic2','topic3']

#-------------onehot------------
for feature in one_hot_feature:
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])

#----------------------
train=data[data.label!=-1]
train_y=train.pop('label')
test=data[data.label==-1]
res=test[['aid','uid']]
test=test.drop('label',axis=1)
enc = OneHotEncoder()

#------create first dimension
enc.fit(data['creativeSize'].values.reshape(-1,1))
train_x=enc.transform(train['creativeSize'].values.reshape(-1,1))
test_x=enc.transform(test['creativeSize'].values.reshape(-1,1))


#-----------save-------------
#joblib.dump(train_x,'one_hot_train_x.pkl')
#joblib.dump(train_y,'train_y.pkl')
#joblib.dump(test_x,'one_hot_test_x.pkl')


#-----------select feature-----------
#kbest=SelectKBest(f_classif,k=100)
percent_10=SelectPercentile(f_classif,percentile=10)
#percent_20=SelectPercentile(f_classif,percentile=20)
#percent_30=SelectPercentile(f_classif,percentile=30)


#onet-hot叠加
for feature in one_hot_feature:
    enc.fit(data[feature].values.reshape(-1, 1))
    train_a=enc.transform(train[feature].values.reshape(-1, 1))
    test_a = enc.transform(test[feature].values.reshape(-1, 1))
    train_x= sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
print('one-hot prepared !')
#--------------------------
cv=CountVectorizer()
for feature in vector_feature:
    cv.fit(data[feature])
    #我感觉可能从稀疏矩阵变成dense.
    train_a = cv.transform(train[feature])
    train_a=percent_10.fit_transform(train_a,train_y)
    
    test_a = cv.transform(test[feature])
    test_a=percent_10.transform(test_a)
    
    train_x = sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
    
   
print('cv prepared !')
