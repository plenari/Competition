# coding: utf-8
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
import requests
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

def get_holiday(years):
    '''获得节假日信息
    
    >>> get_holiday(['2013','2014'])[:1]
    [Timestamp('2013-01-02 00:00:00')]    
    '''
    head={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'}
    holidays=[]
    for year in years:
        url='http://tool.bitefu.net/jiari/?d={}'.format(year)
        r=requests.get(url,headers=head)
        holidays.extend([pd.to_datetime(year+i) for i in eval(r.text)[year]])
    print('get holiday done')
    return holidays

def get_day_balance(file):
    '''读取数据收支表，按天求和

    >>> get_day_balance('')
    
    '''
    
    parse=lambda x:pd.to_datetime(x)
    balance=pd.read_csv(file,parse_dates=True,index_col='report_date',date_parser=parse)
    balance_by_day=balance.groupby('report_date').sum()
    print('get_day_balance done')
    return balance_by_day
    

#数据与日期的关系
def means(df,by):
    '''返回分组后的平均值
    >>>     
    '''
    print('mean done')
    return df.groupby(by).mean()


def plots(y):
    '''画图
    >>> 
    '''
    #月份,删除
    means(y,y.index.month).plot()
    plt.title('by_month')
    plt.show()
    #年份的第几周，删除
    means(y,y.index.weekofyear).plot()
    plt.title('by_weekofyear')
    plt.show()
    #月初
    means(y,y.index.is_month_start).plot(marker='*')
    plt.title('by_start_month')
    plt.show()
    #周几的影响
    means(y,y.index.dayofweek).plot(marker='*')
    plt.title('weekend')
    plt.show()
    #季节
    means(y,y.index.is_quarter_end).plot(marker='*')
    plt.title('by_quarter_end')
    plt.show()
    
    means(y,y.index.is_quarter_start).plot(marker='*')
    plt.title('by_quarter_start')
    plt.show()
    
def h(second,first='index.'):
    '''为了获得日期的属性更加方便
    为了方便需要全局共享h_df变量，此函数才可以运行。
    '''
    print('h start')
    global h_df;df=h_df
    try:
        return eval('df.{}{}'.format(first,second)).reshape(-1,1)
    except:
        return eval('df.{}{}'.format(first,second)).values.reshape(-1,1)
    
def deal_time_x_in(df):
    '''需要df的index是日期的形式,根据是否是周末，季末，月末，。。。。产生一个新的表
    输入get_day_balance的返回结果。
    返回x.shape(427,26)
    >>> 
    '''
    global h_df,holidays
    h_df=df    #
    x=np.hstack([h('weekday'),h('is_quarter_end'),h('is_quarter_start'),h('is_month_end'),h('is_month_start')])
    holiday_in_df=np.array(reg_hol(df.index,holidays)).reshape(-1,1)
    x=np.hstack([x,holiday_in_df]) 
    print('deal time x in  done')
    return x 

def deal_time_x_pre(start='2014-09-01',end='2014-09-30'):
    '''
    >>> 
    '''
    global h_df,holidays
    df=pd.date_range(start,end)#需要预测的时间信息
    h_df=df    #
    x=np.hstack([h('weekday',''),h('is_quarter_end',''),h('is_quarter_start',''),h('is_month_end',''),h('is_month_start','')])
    holiday_in_df=np.array(reg_hol(df,holidays)).reshape(-1,1)
    x=np.hstack([x,holiday_in_df])  
    print('deal time x pre done')
    return x    

def reg_hol(days,rule):
    '''判断days在不在rule里,在就是1，不在就是0
    
    >>> reg_hol(['20130102,20130205'],[20130205])
    [0,1]    
    '''
    print('reg  done')
    return [1 if day in rule else 0 for day in days]

def prep_onehot(X_in,X_pre,degree=5):
    '''用onehot编码X,训练预测的必须用同一个OH编码，所以我同时对两个编码
    X_in:用来训练的X
    X_out:用来预测的X
    >>>     
    '''
    OH=preprocessing.OneHotEncoder(sparse=False)
    X_in2=OH.fit_transform(X_in)
    X_pre2=OH.transform(X_pre)
    #加入时间因素
    X_size=X_in.shape[0]+X_pre.shape[0]
    time=np.linspace(0,1,X_size).reshape(-1,1)
    #多项式
    deg=preprocessing.PolynomialFeatures(degree=degree)
    time_fit=deg.fit_transform(time)
    #组合两个新的X
    X_in3=np.hstack([X_in2,time_fit[:-30,:]])
    X_pre3=np.hstack([X_pre2,time_fit[-30:,:]])
    print('pre onehot done')
    return X_in3,X_pre3,time

def plot_fit(t,x,y,model):
    print('plot_fit start')
    plt.plot(t,y,label='y')
    plt.plot(t,model.predict(x),label='fit')
    plt.legend()
    plt.show()
    plt.close()

def met(fit_y,y):
    '''均方误差而已'''
    print('met done')
    return mean_squared_error(fit_y,y)
    
def cv(model,X,y):
    '''交叉验证而已
    '''
    print('cv done')
    return cross_val_score(model,X,y,cv=10)

def saves(a,b):
    yd=pd.date_range('2014-09-01','2014-09-30')#需要预测的时间信息
    y_result=pd.DataFrame(np.vstack([a,b]).T,columns=['total_purchase_amt','total_redeem_amt'],index=yd.strftime('%Y%m%d'))
    y_result.to_csv('sgd.csv',header=None)
    
    
if __name__=="__main__":
    ffile=r'C:\Users\omf\Desktop\competition\Purchase\Data\user_balance_table.csv'
    holidays=get_holiday(['2013','2014']) 
    balance_by_day=get_day_balance(ffile)
    X_in=deal_time_x_in(balance_by_day)
    X_pre=deal_time_x_pre()
    #分别是输入可以使用的，和预测使用的X
    X_i,X_p,T=prep_onehot(X_in,X_pre)
    #获取两个y
    T_in,T_pre=T[:-30],T[-30:]
    y_buy=balance_by_day['total_purchase_amt'].values.reshape(-1,1)
    y_sell=balance_by_day['total_redeem_amt'].values.reshape(-1,1)
    #构建两个lm
    sgd_buy=SGDRegressor(alpha=1e-5,max_iter=1000,tol=None)
    sgd_sell=SGDRegressor(alpha=1e-5,max_iter=1000,tol=None)
    #拟合
    sgd_buy.fit(X_i,y_buy)
    sgd_sell.fit(X_i,y_sell)
    #plot
    plot_fit(T_in,X_i,y_buy,sgd_buy)
    plot_fit(T_in,X_i,y_sell,sgd_sell)
    #metricx
    print('buy rmse',met(sgd_buy.predict(X_i),y_buy))
    print('sell rmse',met(sgd_sell.predict(X_i),y_sell))
    #cv
    cross_buy=cv(sgd_buy,X_i,y_buy)
    cross_sell=cv(sgd_sell,X_i,y_sell)
    
    #predict
    y_buy_pre=sgd_buy.predict(X_p).reshape(-1,1)
    y_sell_pre=sgd_sell.predict(X_p).reshape(-1,1)
    #save    
    saves(y_buy_pre,y_sell_pre)


