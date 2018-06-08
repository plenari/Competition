# -*- coding: utf-8 -*-
"""
Created on Thu May  3 14:44:13 2018
最重要的是多进程的必须在name==main内部。
@author: omf
"""

# ## duo jin cheng
import pandas as pd
import numpy as np
import multiprocessing
import gc

def cv_svd(i):
    print(i)
    n=np.random.randn(10)
    return  i,n 

objects=['appIdAction','appIdInstall','interest1','interest2',\
                'interest3','interest4','interest5','kw1','kw2','kw3',\
                'topic1','topic2','topic3','os','ct','marriageStatus']

df=pd.DataFrame()
#most important is：
#multiprocessing must in __name__=='__main__'

if __name__=='__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    pool=multiprocessing.Pool(4)    
    result=[]
    for i in objects:
        c=pool.apply(cv_svd,args={i,})
        result.append(c)
    pool.close()
    pool.join()
    for i in result:
        df[i[0]]=i[1]
    del result
    gc.collect()