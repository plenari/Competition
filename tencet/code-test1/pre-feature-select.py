# coding=utf-8
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import  TruncatedSVD
#
#ad_feature=pd.read_csv('data/adFeature.csv')
#user_feature=pd.read_csv('data/userFeature.csv')
#train=pd.read_csv('data/train.csv')
#predict=pd.read_csv('data/test1.csv')
#
#train.loc[train['label']==-1,'label']=0
#predict['label']=-1
#
#data=pd.concat([train,predict])
#data=pd.merge(data,ad_feature,on='aid',how='left')
#data=pd.merge(data,user_feature,on='uid',how='left')
#data=data.fillna('-1')
#data.to_csv('data_after_merge.csv',index=None)

#--------------read data--------
'''
train's label is 0 or 1
test's label is -1
'''
data=pd.read_csv('data_after_merge.csv',index_col=None)

#-----------split train test----------
train_x=data[data.label!=-1]
train_y=train_x['label']
train_x=train_x.drop('label',axis=1)
test_x=data[data.label==-1]
test_x=test_x.drop('label',axis=1)
res=test_x[['aid','uid']]

#---------------plot hist------
def hists(name,data):    
    fig=plt.figure()
    plt.hist(data)
    plt.title(name)
    fig.savefig('train_x_%s.png'%name)
    fig.clear()
numbers=[]
objects=[]    
for i in train_x:
    if train_x[i].dtype=='object':
        objects.append(i)
    else:
        numbers.append(i)
        hists(i,train_x[i])
#--delete uid
numbers.remove('uid')
numbers.remove('aid')
#--------------objects---------------- 
'''
objects=['marriageStatus', 'interest1', 'interest2', 'interest3',
 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2',
 'topic3', 'appIdInstall', 'appIdAction', 'ct', 'os']

 numbers=['aid', 'advertiserId', 'campaignId', 'creativeId',
 'creativeSize', 'adCategoryId', 'productId', 'productType',
 'age', 'gender', 'education', 'consumptionAbility', 'LBS',
 'carrier', 'house']
 '''   


train_xi=train_x[numbers]
test_xi=test_x[numbers]
#------------decompostion of object-----------
cv=CountVectorizer(token_pattern='(?u)\\b\\w+\\b')

for feature in objects:    
    train_tmp=cv.fit_transform(train_x[feature])    
    test_tmp = cv.transform(test_x[feature])
    #--------decompostion-----
    dimension=np.int(np.ceil(test_tmp.shape[1]*0.001))
    svd=TruncatedSVD(n_components=dimension,n_iter=5)    
    train_a=svd.fit_transform(train_tmp)
    test_a=svd.transform(test_tmp)
    train_xi = sparse.hstack((train_xi, train_a))
    test_xi = sparse.hstack((test_xi, test_a))
    print('feature hstack done!')
print('cv prepared !')

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,row=array.row,
             col =array.col, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.coo_matrix((  loader['data'], (loader['row'], loader['col'])),
                         shape = loader['shape'])

save_sparse_csr('train_x',train_xi)
save_sparse_csr('test_x',test_xi)
res.to_csv('res.csv',index=None)
train_y.to_csv('train_y.csv',index=None)