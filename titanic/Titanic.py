# coding: utf-8

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score,precision_score,precision_recall_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier


class pre:
    '''前期数据处理'''
    def __init__(self):
        '''初始化一些训练和预测共同的方法'''
        self.Sex=preprocessing.LabelBinarizer()
        self.Em=preprocessing.LabelEncoder()
        self.Cabin=preprocessing.LabelEncoder()
        self.Age=preprocessing.Imputer(missing_values='NaN', strategy='mean',axis=1)
        self.all=preprocessing.StandardScaler()
        self.kong=preprocessing.Imputer(missing_values='NaN', strategy='mean',axis=1)
    
    def str_(self,):
        '''处理字符串'''
        self.df.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
        self.df['Cabin']=[str(i)[0] for i in self.df['Cabin']]
        self.df['Embarked']=self.df['Embarked'].fillna('N')
        
    def fit_transform(self,df):
        '''返回train处理后的自变量和因变量'''
        self.df=df.copy()
        self.str_()
        self.df['Sex'] = self.Sex.fit_transform(self.df['Sex'])   
        self.df['Embarked'] = self.Em.fit_transform(self.df['Embarked'])
        self.df['Cabin'] = self.Cabin.fit_transform(self.df['Cabin'])
        self.df['Age'] = self.Age.fit_transform(self.df['Age'].values.reshape(1,-1)).flatten()
        X,y=self.df.iloc[:,1:],self.df.iloc[:,0]
        X=self.kong.fit_transform(X)
        X=self.all.fit_transform(X)
        return X,y
    
    def transform(self,df):
        '''返回test的因变量'''
        self.df=df.copy()
        self.str_()
        self.df['Sex'] = self.Sex.transform(self.df['Sex'])   
        self.df['Embarked'] = self.Em.transform(self.df['Embarked'])
        self.df['Cabin'] = self.Cabin.transform(self.df['Cabin'])
        self.df['Age'] = self.Age.transform(self.df['Age'].values.reshape(1,-1)).flatten() 
        X=self.kong.fit_transform(self.df.as_matrix())
        X=self.all.transform(X)
        return X
    
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    
def gridsearch(clf,X,y):
    '''网格搜索
    clf:模型
    X:y,
    '''        
    param={'n_estimators':np.arange(10,30,5),'max_depth':np.arange(5,20,5)}    
    cv=GridSearchCV(clf,param_grid=param,return_train_score=True)    
    cv.fit(X,y)    
    print('交叉验证最好的得分',cv.best_score_)
    return cv

def random_cv(clf,X,y):
    param={'n_estimators':np.arange(10,30,5),'max_depth':np.arange(5,20,5)} 
    rcv=RandomizedSearchCV(clf,param_distributions=param,return_train_score=True)
    rcv.fit(X,y)
    return rcv
def DTC(X,y):
    # ### 决策树  
    clft=DecisionTreeClassifier()
    clft.fit(X,y)    
    #cross_val_score(clft,X,y,cv=10)
    return clft

def ana():   
    '''
    把一些零碎的东西结合到这里。    
    '''
    df=pd.read_csv('train.csv')
    test=pd.read_csv('test.csv')
    tr=pre()
    X,y=tr.fit_transform(df)
    X_pre=tr.transform(test)
    # #### 拟合
    clf=RandomForestClassifier(n_estimators=50)
    clf.fit(X,y)
    print('随机森林得分：',np.max(cross_val_score(clf,X,y,cv=10)))

    #查准率（Precision）预测为正类的样本有多少比例的样本是真的正类
    #查全率（recall）Recall就是所有真正的正类样本有多少比例被预测为正类，
    print('recall:{},precision:{}'.format(recall_score(y,clf.predict(X)),precision_score(y,clf.predict(X))))
    print('f1_socre:{}'.format(f1_score(y,clf.predict(X))))
    #precisions, recalls, thresholds=precision_recall_curve(y,clf.predict(X))
    #plot_precision_recall_vs_threshold(precisions,recalls,thresholds)
    #plt.show()
    #保存最好模型预测结果
    cv=gridsearch(clf,X,y)
    y_pre=cv.best_estimator_.predict(X_pre)    
    pd.DataFrame(np.array([test.PassengerId,y_pre]).T,columns=['PassengerID','Survived']).to_csv('RandomForestClassifier.csv',index=False)

    # #### 均方误差    
    print('mse:',mean_squared_error(y,clf.predict(X)))
ana()