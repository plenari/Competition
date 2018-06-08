# -*- coding: utf-8 -*-
"""
Created on Sat May  5 15:09:07 2018
@author:Plenari
@email:shengjiex@qq.com
"""
import pandas as pd
import numpy as np  
import re
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest 
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn import preprocessing
import gc

# Load in the train and test datasets
train = pd.read_csv('/data/train.csv')
test_x = pd.read_csv('/data/test.csv')
train_x=train.drop('Survived',axis=1)
# Store our passenger ID for easy access
res = test_x['PassengerId']
train_y=train.Survived
data=pd.concat([train_x,test_x],axis=0)

#get ----differenct type data-------
head_data=data.head(2)
objects=head_data.describe(include='object').columns
numbers=np.array(list(set(head_data.columns.values)-set(objects)))
del head_data
gc.collect()
#['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']

'''
1. number 类型：
    'Fare', 'SibSp', 'Age',  'Parch', 'Pclass'
    Fare: 
    先处理缺失值
     

2. object 类型：
    array(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], dtype=object))
    Ticket 暂不处理
    Cabin 缺失值较少用众数填充
    EMbarked 缺失较多，缺失成一类
    
3. 先处理object，然后处理离散变量，最后连续变量
'''
def new_fare(data,train,test,col):
    '''
    data is train and test
    col is the columns name
    '''
    #把票价分成三类lt25,25-200,bt200
    clas=[25,200]
    train_=np.searchsorted(clas,train[col])
    test_=np.searchsorted(clas,test[col])
    return train_,test_
train_x['N_Fare'],test_x['N_Fare']=new_fare(data,train_x,test_x,'Fare')

# deal with parch----------
def new_parch(data,train,test,col):
    '''
    data is train and test
    col is the columns name
    '''
    #把parch分成三类0,1-2,bt3
    clas=[0.3,2.3]
    train_=np.searchsorted(clas,train[col])
    test_=np.searchsorted(clas,test[col])
    return train_,test_

train_x['N_Parch'],test_x['N_Parch']=new_parch(data,train_x,test_x,'Parch')

# deal with parch----------
def new_sibsp(data,train,test,col):
    '''
    data is train and test
    col is the columns name
    '''
    #把sibsp分成三类0,1,bt2
    clas=[0.3,1.3]
    train_=np.searchsorted(clas,train[col])
    test_=np.searchsorted(clas,test[col])
    return train_,test_

train_x['N_SibSp'],test_x['N_SibSp']=new_sibsp(data,train_x,test_x,'SibSp')

# deal with Pclass-------------------
train_x['N_Pclass'],test_x['N_Pclass']=train_x['Pclass'],test_x['Pclass']
    
#---deal sex----
def new_sex(data,train,test,col):
    '''
    data is train and test
    col is the columns name
    '''
    #LabelEncode
    Le=preprocessing.LabelEncoder()
    Le.fit(data[col])
    train_=Le.transform(train[col])
    test_=Le.transform(test[col])
    return train_,test_
train_x['N_Sex'],test_x['N_Sex']=new_sex(data,train_x,test_x,'Sex')

#-------deal with name------------
def new_name(data,train,test,col):
    '''
    data is train and test
    col is the columns name
    '''
    #提取职称
    compiles=re.compile(r'(M[A-Za-z]*)\.')
    train_=train[col].apply(lambda x :re.findall(compiles,x)[0] if len(re.findall(compiles,x))==1 else 'Mr')
    test_=test[col].apply(lambda x :re.findall(compiles,x)[0] if len(re.findall(compiles,x))==1 else 'Mr')   
    Le=preprocessing.LabelEncoder()
    Le.fit(list(set(test_) | set(train_)))
    train_=Le.transform(train_)
    test_=Le.transform(test_)
    return train_,test_
train_x['N_Name'],test_x['N_Name']=new_name(data,train_x,test_x,'Name')

#----------------cabin-----------
def new_cabin(data,train,test,col):
    '''
    data is train and test
    col is the columns name
    '''    
    #仓位首字母，以及没有仓位
    data_=data[col].apply(lambda x:str(x)[0])
    train_=train[col].apply(lambda x:str(x)[0])
    test_=test[col].apply(lambda x:str(x)[0])
    Le=preprocessing.LabelEncoder()
    Le.fit(data_)
    train_=Le.transform(train_)
    test_=Le.transform(test_)
    return train_,test_
train_x['N_Cabin'],test_x['N_Cabin']=new_cabin(data,train_x,test_x,'Cabin')
#------deal with embarked------------
def new_embarked(data,train,test,col):
    '''
    data is train and test
    col is the columns name
    '''    
    #用众数填充
    mode=data[col].value_counts().sort_values(ascending=False).index[0]
    data_=data[col].fillna(mode)
    train_=train[col].fillna(mode)
    test_=test[col].fillna(mode)

    Le=preprocessing.LabelEncoder()
    Le.fit(data_)
    train_=Le.transform(train_)
    test_=Le.transform(test_)
    return train_,test_
train_x['N_Embarked'],test_x['N_Embarked']=new_embarked(data,train_x,test_x,'Embarked')

#deal with age-----------
def new_age(data,train,test,col):
    '''
    data is train and test
    col is the columns name
    20% is null
    '''   
    #lt5,5-15,15-35,35-50,bt50
    clas=[5,15,35,50]    
    train_=np.searchsorted(clas,train[col])
    test_=np.searchsorted(clas,test[col])
    return train_,test_
train_x['N_Age'],test_x['N_Age']=new_age(data,train_x,test_x,'Age')

new_col=[i for i in test_x.columns if i.startswith('N_')]

train_x=train_x[new_col]
test_x=test_x[new_col]
print(train_x.shape,test_x.shape,res.shape,train_y.shape)
train_x.to_csv('train_x.csv')
test_x.to_csv('test_x.csv')
np.savez('res_label.npz',res=res.values,label=train_y.values)
