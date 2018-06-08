# coding: utf-8
import pandas as pd
import os 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.decomposition import  TruncatedSVD
import gc

# ## duo jin cheng
import multiprocessing
#---------merge data-------
ad_feature=pd.read_csv('data/adFeature.csv')
user_feature=pd.read_csv('data/userFeature.csv')
train=pd.read_csv('data/train.csv')
test1=pd.read_csv('data/test1.csv')
#------result--------
res=test1.copy()

train.loc[train['label']==-1,'label']=0
test1['label']=-1
#
data=pd.concat([train,test1])
data=pd.merge(data,ad_feature,on='aid',how='left')
data=pd.merge(data,user_feature,on='uid',how='left')
#fill na
# ## deal test'dtype  is int or object
numbers=[]
objects=[]    
data_head=data.head()
for i in data_head:
    if data_head[i].dtype=='object':
        objects.append(i)
        data[i]=data[i].fillna('-1')
    else:
        data[i]=data[i].fillna(-1)
        numbers.append(i)
del data_head
gc.collect()
       
aid_uid=['aid','uid','label']
for i in aid_uid:
    numbers.remove(i)

#------------del -------
del test1,train,user_feature,ad_feature 
gc.collect()

#------------
train=data[data.label!=-1]
test=data[data.label==-1]
#---------data without label
train_y=data.pop('label')
train_x=train
test.pop('label')

'''
numbers=['LBS','age','carrier','consumptionAbility','education',\
                 'campaignId', 'creativeId','adCategoryId',\
                 'productId', 'productType']
objects=['appIdAction','appIdInstall','interest1','interest2',\
                'interest3','interest4','interest5','kw1','kw2','kw3',\
                'topic1','topic2','topic3','os','ct','marriageStatus']
'''

train_res=train_x[numbers]
test_res=test[numbers]


def cv_svd(feature):
    #global test_obj_dec,train_obj_dec
    cv=CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
    train_tmp=cv.fit_transform(train_x[feature])    
    test_tmp = cv.transform(test[feature])
    #--------decompostion-----
    #dimension=np.int(np.ceil(test_tmp.shape[1]*0.001))
    svd=TruncatedSVD(n_components=1,n_iter=5)    
    train_a=svd.fit_transform(train_tmp)
    test_a=svd.transform(test_tmp)
    print('{} is  done!'.format(feature))
    return train_a,test_a,feature

if __name__=='__main__':
    pool=multiprocessing.Pool(4)
    result=[]
    for i in objects:
        c=pool.apply(cv_svd,args={i,})
        result.append(c)
    pool.close()
    pool.join()
    
    for res in result:
        train_res[res[2]]=res[0]
        test_res[res[2]]=res[1]
    
    del result
    gc.collect()
    #---------save-----
    train_res.to_csv('train_x.csv')
    test_res.to_csv('test_x.csv')
    train_y.to_csv('train_y.csv')
    res.to_csv('test.csv')
#--------save------------------
'''np.savez('train_test_data_obj_type_after_cv_dec.npz', \
         train_obj_dec=train_obj_dec,test_obj_dec=test_obj_dec)



def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,row=array.row,
             col =array.col, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.coo_matrix((  loader['data'], (loader['row'], loader['col'])),
                         shape = loader['shape'])
'''