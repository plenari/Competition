# -*- coding: utf-8 -*-
"""
Created on Thu May  3 16:12:25 2018
@email:shengjiex@qq.com
@author: Plenari
"""
import pandas as pd
import numpy as np
import datetime
from math import radians, cos, sin, asin, sqrt
from collections import Counter 
import os
import warnings
from sklearn.neighbors import KNeighborsClassifier

__version__=2.0
'''

'''
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
df=pd.read_csv('jing_wei.csv',index_col=0)
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(df[['j','w']].values,df.index.values)

def haversine(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    return    float
    
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000

def Height(user):
    '''
    计算四种海拔高度的概率
    ['sm500,'sm1000','sm1500','bt1500']
    '''
    name=['sm500','sm1000','sm1500','bt1500']
    clas_=[500,1000,1500]
    sort_height=np.searchsorted(clas_,user.HEIGHT)
    res_=[]
    lenofdata=len(sort_height)
    count=Counter(sort_height)    
    for i in range(len(clas_)+1):
        res_.append(count[i]/lenofdata) 
    return res_,name

def Hour_24(user):
    '''
    groupby_HOUR
    输入单个用户的全部数据
    计算单个用户的开车习惯分布
    '''
    name=['h0','h1','h2','h3','h4','h5','h6','h7','h8',\
          'h9','h10','h11','h12','h13','h14','h15','h16',\
          'h17','h18','h19','h20','h21','h22','h23']
    hour24=pd.DataFrame(np.zeros(24),columns=['hour'])
    time_last=user.hour.value_counts().sort_index()
    res_=time_last/time_last.sum()
    hour24.update(res_)
    return list(hour24.hour),name

def nc(flag,clas_,list_):
    '''
    name create
    flag :前缀
    clas_:类别
    list_:列表
    '''
    for i in clas_:        
        list_.append('st-'+flag+str(i))
    list_.append('bt-'+flag+str(i))
    return list_

def Speed(user):
    '''
    user dtype DataFrame 
    columns SPEED
    
    return :list
    ['mean' ,'std','max',]
    2.计算速度在cla_之间的概率
    '''
    name=['mean' ,'std','max']
    s=user.SPEED   
    clas_=[0,18,28,38,60,80,120]
    speed=np.searchsorted(clas_,user.SPEED)
    speed=after_searchsorted(speed,clas_)
    name=nc('speed',clas_,name)
    result=[s.mean(),s.std(),s.max()]
    result.extend(speed)
    return result,name

def Callstate(user):
    '''
    对于每个用户电话状态是1，2，3的概率
    '''
    name=[]
    clas_=[0.5,1.5,2.5,3.5]
    result=after_searchsorted(user.CALLSTATE,clas_)
    name=nc('call',clas_,name)
    return  result,name

def after_searchsorted(data,clas_):
    '''
    searchsorted 之后需要算各个类别的概率
    所以用这个方便处理各个概率，
    data:得到插入排名后的数据
    clas_:类别， 因为数据可能不全，只能靠类别获得类别数
    return: list
    '''
    res_=[]
    lenofdata=len(data)
    count=Counter(data)    
    for i in range(len(clas_)+1):
        res_.append(count[i]/lenofdata) 
    return res_

def Direction(user):
    '''
    一般大路应该都是正南正北，正东正西
    用来计算与0,90,180,360，的夹角小于0.01
    return float 
    'direction'
    在大路上开的概率
    '''
    name=['direction']
    direct=user.DIRECTION/90#direction
    direct=np.abs(direct-np.round(direct))
    return [(direct<0.01).mean()],name

#通过行程进行分组，然后计算每次行程的运行距离，时间，在早中晚高峰的情况
def Trip_id(user):
    '''    
    输入一个user的数据包含DataFrame()    
    
    包含：TIME,hour
    返回这个用户在移动距离的平均值    
    return list
    ['distance','sm30','sm60','sm120','sm240','bt240','dayofweek','isweekday','morning','afternoon','noon','nightdriver']
    '''
    name=['distance_mean','distance_std','acceleration_mean','acceleration_std','city_in','sm30','sm60','sm120','sm240','bt240','dayofweek','isweekday',
          'morning','afternoon','noon','nightdriver']
    #保存最终结果。
    result_=[]
    user_g=user.groupby('TRIP_ID')
    #分组计算平均移动距离,所在城市的众数
    city_trip_end=[]
    distance_=[]
    acceleration_=[]
    for i,trip in user_g:
        '''按照tripid分组的数据 '''
        #获得行程开始结束的经纬度
        lon1,lon2= trip['LONGITUDE'].iloc[[0,-1]]
        lat1,lat2=trip['LATITUDE'].iloc[[0,-1]]
        #预测最后行程最后结束时，所在的城市。
        city_trip_end.append(knn.predict(np.array([lon2,lat2]).reshape(-1,2))[0])
        #计算移动距离
        res_i=haversine(lon1, lat1, lon2, lat2)
        distance_.append(res_i)
        
        #计算加速度
        speed_=trip.SPEED.values[-2:]
        if speed_.size>1:
            acceleration_.append(speed_[-1]-speed_[-2])
        
    #添加平均移动距离到结果
    result_.extend([np.mean(distance_),np.std(distance_)])
    #加速度
    result_.extend([np.mean(acceleration_),np.std(acceleration_)])
    
    #所在城市的众数
    result_.append(pd.Series(city_trip_end).mode().loc[0])
    
    
    #开车时长，简单用计数处理,分钟处理,分别为半个小时，一个小时，两个小时，4个小时的概率
    
    drive_last_time=user_g.hour.count()
    clas_=[30,60,120,240]
    clas_time_last=np.searchsorted(clas_,drive_last_time)
    lenofdata=clas_time_last.size
    count=Counter(clas_time_last)
    for i in range(len(clas_)+1):
        #连续驾驶时长段，所占的概率
        result_.append(count[i]/lenofdata)        
        
    #分组计算行程终止时星期几，以及周末的概率
    user_trip_tail=user_g.apply(lambda x:x.tail(1))   
    user_trip_tail['dayofweek']=user_trip_tail.TIME.apply(lambda x:x.dayofweek)
    user_trip_tail['isweekend']=user_trip_tail.dayofweek.apply(lambda x:1 if x  in [5,6] else 0)
    #-------早晚中高峰
    user_trip_tail['morning']=user_trip_tail.hour.apply(lambda x: 1 if x in [7,8,9] else 0)
    user_trip_tail['afternoon']=user_trip_tail.hour.apply(lambda x: 1 if x in [17,18,19] else 0)
    user_trip_tail['noon']=user_trip_tail.hour.apply(lambda x: 1 if x in [11,12,13] else 0)
    time_=user_trip_tail[['dayofweek','isweekend','morning','afternoon','noon']].mean()
    
    #---save data
    result_.extend(time_)
    #day or night白天还是晚上开车的概率，白天为1
    night_driver_car=user.hour.apply(lambda x :x<7 and x>19).mean()
    #'distance','sm30','sm60','sm120','sm240','bt240','dayofweek','isweekend','morning','afternoon','noon','daydriver'
    result_.append(night_driver_car)  
    assert len(result_)==len(name) ,'长度不一样'
    return result_ ,name

# path
path_train = "/data/dm/train.csv"  # 训练文件路径
path_test = "/data/dm/test.csv"  # 测试文件路径

def deal_train():
    if os.path.isfile(r'G:\pingan\data\dm\train.csv'):
        train=pd.read_csv(r'G:\pingan\data\dm\train.csv')
    else:
        train=pd.read_csv(path_train)
    
    print('TRAIN.Y==0 mean:',(train.Y==0.0).mean(),)
    print('mean,std:',train.Y.mean(),train.Y.std())
    print('fen duan:\n',pd.cut(train.Y,10).value_counts())
    train['TIME']=train.TIME.apply(lambda x:datetime.datetime.fromtimestamp(x))
    #deal train.Y 极值限制到0 ymean +2*ystd
    train_y_b_0=train.Y[train.Y>0]
    y_mean,y_std=train_y_b_0.mean(),train_y_b_0.std()
    print('train.Y>0 mean ,std',y_mean,y_std)
    #train['Y']=np.clip(train.Y,0,y_mean+2*y_std)
    
    ##train
    N_train=[]
    columns=[]
    for i,user in train.groupby('TERMINALNO'): 
        user_data=[]
        user_data.append(user.Y.iloc[0])
        user['hour']=user.TIME.apply(lambda x:x.hour)
        #* 计算所有行程的时间分布
        hour_24=Hour_24(user)
        user_data.extend(hour_24[0])
        #* 计算每次行程的平均移动距离
        trip_id=Trip_id(user)
        user_data.extend(trip_id[0])
        #SPEED
        speed=Speed(user)
        user_data.extend(speed[0])
        #电话状态
        callstate=Callstate(user)
        user_data.extend(callstate[0])
        #方向
        direction=Direction(user)
        user_data.extend(direction[0])
        if i==1:
            columns.append('Y')
            columns.extend(hour_24[1])
            columns.extend(trip_id[1])
            columns.extend(speed[1])
            columns.extend(callstate[1])
            columns.extend(direction[1])        
        N_train.append(user_data)
    train=pd.DataFrame(N_train,columns=columns)
    train_y=train.pop('Y').values
    return train,train_y


def deal_test():
    if os.path.isfile(r'G:\pingan\data\dm\test.csv'):
        train=pd.read_csv(r'G:\pingan\data\dm\test.csv')
    else:   
        train=pd.read_csv(path_test)
    train['TIME']=train.TIME.apply(lambda x:datetime.datetime.fromtimestamp(x))
       
    ##train
    N_train=[]
    columns=[]
    for i,user in train.groupby('TERMINALNO'): 
        user_data=[]
        user_data.append(user.TERMINALNO.iloc[0])
        user['hour']=user.TIME.apply(lambda x:x.hour)
        #* 计算所有行程的时间分布
        hour_24=Hour_24(user)
        user_data.extend(hour_24[0])
        #* 计算每次行程的平均移动距离
        trip_id=Trip_id(user)
        user_data.extend(trip_id[0])
        #SPEED
        speed=Speed(user)
        user_data.extend(speed[0])
        #电话状态
        callstate=Callstate(user)
        user_data.extend(callstate[0])
        #方向
        direction=Direction(user)
        user_data.extend(direction[0])
        if i==1:
            columns.append('ID')
            columns.extend(hour_24[1])
            columns.extend(trip_id[1])
            columns.extend(speed[1])
            columns.extend(callstate[1])
            columns.extend(direction[1])        
        N_train.append(user_data)
    train=pd.DataFrame(N_train,columns=columns)
    train_Id=train.pop('ID').values
    return train,train_Id
	
	
	
	
	