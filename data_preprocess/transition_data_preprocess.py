# -*- coding: utf-8 -*-
'''
Created on 2018年11月27日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import pandas as pd
import numpy
# transition_data=pd.read_csv('../data/transition_train.csv')
# print(transition_data.head(10))
# dataset=pd.get_dummies(transition_data,columns=['o_city_code','o_district_code','d_city_code','d_district_code'])
# dataset.to_csv('./data_onehot.csv') 
dataset=pd.read_csv('./data_onehot.csv')  

print(dataset.columns.values.tolist())
dataset=dataset.drop(columns=['Unnamed: 0'])
print(dataset['date_dt'][:10,],len(dataset))
dataset=dataset.drop_duplicates()
print(dataset['date_dt'][:10,],len(dataset))

        