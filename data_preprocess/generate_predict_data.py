# -*- coding: utf-8 -*-
'''
Created on 2018年12月6日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('../data/transition_train.csv')
pre_data=pd.DataFrame()
date_dt=[]
series_num=[]
for i in range(0,15):
    date_dt.append('2018/03/'+str(2+i))
date_dt1=[]
ser_no1=[]
d_city_code=[]
d_district_code=[]
o_city_code=[]
o_district_code=[]
week_no1=[]
ser_no=285
week_no=40
for i in date_dt:
    for j in dataset['d_city_code'].unique():
        for k in dataset['d_district_code'].unique():
            for m in dataset['o_city_code'].unique():
                for n in dataset['o_district_code'].unique():
                    if k!=n:
                        if ser_no<=286:
                            week_no=40
                        else:
                            week_no=(ser_no-286)/7+41
                        date_dt1.append(i)
                        ser_no1.append(ser_no)
                        d_city_code.append(j)
                        d_district_code.append(k)
                        o_city_code.append(m)
                        o_district_code.append(n)
                        week_no1.append(week_no)
                    else:
                        continue
    ser_no+=1
pre_data['date_dt']=date_dt1      
pre_data['series_num']=ser_no1   
pre_data['d_city_code'] =d_city_code
pre_data['d_district_code'] =d_district_code
pre_data['o_city_code'] =o_city_code
pre_data['o_district_code'] =o_district_code
pre_data['week_no']=week_no1
pre_data.to_csv('./pre_data.csv')                   
                    
                    
                    
                    
        
    
    
    