# -*- coding: utf-8 -*-
"""
Created on Thu May  3 16:12:25 2018

@author: omf
"""
import pandas as pd
import numpy as np
import datetime
from math import radians, cos, sin, asin, sqrt
from collections import Counter 
import os

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

def Speed(user):
    '''
    user dtype DataFrame 
    columns SPEED
    
    return :list
    ['mean' ,'std','max','spped_30_40']
    2.计算速度在30-40之间的概率
    '''
    name=['mean' ,'std','max','spped_30_40']
    s=user.SPEED   
    speed_30_40=((s>30) & (s<40)).mean()
    return [s.mean(),s.std(),s.max(),speed_30_40],name

def Callstate(user):
    '''
    对于每个用户电话状态是1，2，3的概率
    '''
    name=['call']
    s=user.CALLSTATE
    res_=[1 for i in s if i in [1,2,3] ]
    return  [len(res_)/s.size],name

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
    name=['distance','sm30','sm60','sm120','sm240','bt240','dayofweek','isweekday',\
          'morning','afternoon','noon','nightdriver']
    result_=[]
    distance_=[]
    user_g=user.groupby('TRIP_ID')
    #分组计算平均移动距离
    for i,trip in user_g:
        '''按照tripid分组的数据 '''
        lon1,lon2= trip['LONGITUDE'].iloc[[0,-1]]
        lat1,lat2=trip['LATITUDE'].iloc[[0,-1]]
        res_i=haversine(lon1, lat1, lon2, lat2)
        distance_.append(res_i)
    result_.append(np.mean(distance_))
    
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
    return result_ ,name 

# path
path_train = "/data/dm/train.csv"  # 训练文件路径
path_test = "/data/dm/test.csv"  # 测试文件路径

def deal_train():
    if os.path.isfile(r'G:\pingan\data\dm\train.csv'):
        train=pd.read_csv(r'G:\pingan\data\dm\train.csv')
    else:
        train=pd.read_csv(path_train)
    print((train.Y==0.0).mean())
    train['TIME']=train.TIME.apply(lambda x:datetime.datetime.fromtimestamp(x))
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
	
	
	
	
	