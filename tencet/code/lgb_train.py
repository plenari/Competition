# coding=utf-8
'''
1. 这个准备用来写一个可以持续持续更新的模型
持续更新的模型是用dataset来喂数据的，

2. 更改countvector的的方法
3. 加入aid特征，用predict的复制备份当做res
'''
import pandas as pd
import lightgbm as lgb
#from sklearn.model_selection import ParameterGrid
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import time
import pickle
#
ad_feature=pd.read_csv('data/adFeature.csv')
user_feature=pd.read_csv('data/userFeature.csv')
train=pd.read_csv('data/train.csv')
predict=pd.read_csv('data/test1.csv')
res=predict.copy()
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
                 'campaignId', 'creativeId','adCategoryId', 'aid',\
                 'productId', 'productType']
vector_feature=['appIdAction','appIdInstall','interest1','interest2',\
                'interest3','interest4','interest5','kw1','kw2','kw3',\
                'topic1','topic2','topic3','os','ct','marriageStatus']
for feature in one_hot_feature:
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])

train=data[data.label!=-1]
train_y=train.pop('label')
test=data[data.label==-1]
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

cv=CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
for feature in vector_feature:
    cv.fit(data[feature])
    train_a = cv.transform(train[feature])
    test_a = cv.transform(test[feature])
    train_x = sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
print('cv prepared !')

#----------------Dataset-----------------
# if you want to re-use data, remember to set free_raw_data=False
lgb_train = lgb.Dataset(train_x, train_y,\
            free_raw_data=False)

#--------------set paras-------------
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 40,
    'learning_rate': 0.10,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 30,
    'verbose': 0,
    'reg_lambda':0.9,
    'early_stopping_rounds':100,  
    'subsample':0.8,
    'subsample_freq':30,   
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


def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])
'''

#----------------save model---------------------------
now=time.strftime('%m-%d-%H-%M',time.localtime())
try:
    estimator.save_model('model-{}.txt'.format(now))
    with open('test_x.pkl','wb') as f:
        pickle.dump(test_x,f)
    with open('res.pkl','wb') as f:
        pickle.dump(res,f)
except Exception as e:
    print('errors:',e)

#--------------------prediction-----------------------
res['score'] = estimator.predict(test_x)
res['score'] = res['score'].apply(lambda x: float('%.8f' % x))
res.to_csv('data/submission-{}.csv'.format(now), index=False)

