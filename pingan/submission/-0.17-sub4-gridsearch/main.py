import pandas as pd
import numpy as np  
import lightgbm as lgb
import datetime
from prepro import deal_train,deal_test
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

start_all0 = datetime.datetime.now()

test_x,testID=deal_test()
train_x,train_y=deal_train()
train_x.fillna(0)
test_x.fillna(0)

print("*******DATA*******:{:.2f} minute".format((datetime.datetime.now()-start_all0).seconds/60))

start_all = datetime.datetime.now()
paras={'min_data':[1],
       'min_data_in_leaf':[1],
       'min_sum_hessian_in_leaf' :[11],
       'num_leaves':[5,9],
       'learning_rate':[0.01], 
       'n_estimators':[720],    
       'bagging_fraction':[0.8,0.5],
       'bagging_freq':[5,10], 
       'feature_fraction':[0.2319,0.5],
       'boosting_type':['rf','gbdt'],
       'objective':['regression','quantile',]
           }

#split data
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
#gridsearch
def ParameterGrid_run(paras,train_x,train_y,test_x,test_y,first=1):
    '''
    paras:参数
    train_x:
    train_y:
    '''
    PG=ParameterGrid(paras)
    print('there are {} for train'.format(len(list(PG))))
    res=[]
    #clfs=[]
    clf_best=''
    best_l1=np.inf
    for i,para in enumerate(PG):        
        try:
            clf=lgb.LGBMRegressor(max_bin = 55, 
                                  feature_fraction_seed=9, 
                                  bagging_seed=9,**para)

            clf.fit(X_train,y_train,eval_set=[(test_x,test_y,)],early_stopping_rounds=50,verbose=False,eval_metric='l1')
            para['score']=clf.best_score_.get('valid_0')['l1']
            res.append(para)
            if para['score']<best_l1:
                clf_best=clf
            #clfs.append(clf.booster_)
            if i%10==0:
                print(i,para['score'])
        except Exception as e:
            print(e)
    #还缺一个保存结果的文件。比如保存得分前十的结果和模型    
    res=pd.DataFrame(res)
    res=res.sort_values('score',ascending=True)   
    
    return res,clf_best

    
res,gbm=ParameterGrid_run(paras,X_train,y_train,X_test,y_test)
col=['bagging_fraction', 'bagging_freq', 'boosting_type', 'feature_fraction', 'num_leaves', 'objective','score']
print('*'*20,'PARAS','*'*20)
print(res[col].head())


#--save model ----------
#gbm.booster_.save_model('model.txt')
#print('save done')
# output result
result = pd.DataFrame(testID,columns=['Id'])
result['Pred'] = gbm.predict(test_x)

result.to_csv('./model/result.csv',header=True,index=False)

print("********FIT********:{:.2f} minute".format((datetime.datetime.now()-start_all).seconds/60))