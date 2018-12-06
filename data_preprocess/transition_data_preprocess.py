# -*- coding: utf-8 -*-
'''
Created on 2018年11月27日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error   #均方误差回归损失
# transition_data=pd.read_csv('../data/transition_train.csv')
# print(transition_data.head(10))
# dataset=pd.get_dummies(transition_data,columns=['o_city_code','o_district_code','d_city_code','d_district_code'])
# dataset.to_csv('./data_onehot.csv') 
# dataset=pd.read_csv('./data_onehot.csv')  

# print(dataset.columns.values.tolist())
# dataset=dataset.drop(columns=['Unnamed: 0'])
# print(dataset['date_dt'][:10,],len(dataset))
# dataset=dataset.drop_duplicates()
# print(dataset['date_dt'][:10,],len(dataset))
# a={}
# j=1
# for i in dataset['date_dt'].unique():
#     a[i]=j
#     j+=1
# series_num=[]
# for i in dataset['date_dt']:
#     series_num.append(a[i])
# dataset['series_num']=series_num
# dataset.to_csv('./data_onehot_2.csv')
# dataset=pd.read_csv('./data_onehot_1_1.csv')
# print(dataset.columns.values.tolist())
# week_no=[]
# for i in dataset['week']:
#     week_no.append(str(i).replace('周', ''))
# dataset['week_no']=week_no
# dataset=dataset.drop(columns=['date_dt','week'])
# dataset=pd.get_dummies(dataset,columns=['work_day'])
# dataset=dataset.astype(float)
# dataset.to_csv('./train_trans.csv')
dataset_1=pd.read_csv('./train_trans_1.csv')
label=dataset_1['cnt']
feature_set=dataset_1.drop(columns=['cnt'])
feature_name=feature_set.columns.values.tolist()
print(len(feature_name))
x_train,x_test,y_train,y_test=train_test_split(feature_set,label,random_state=1)
clf=tree.DecisionTreeRegressor()#决策树回归模型
clf=clf.fit(x_train,y_train)
# y_importances=clf.feature_importances_
# print(y_importances)
# x_importances=feature_name
# y_pos=np.arange(len(x_importances))
# #横向柱状图
# plt.barh(y_pos,y_importances,align='center')
# plt.yticks(y_pos,x_importances)
# plt.xlabel('importances')
# plt.xlim(0,1)
# plt.title('feature importance')
# plt.show()
# from sklearn.linear_model import LinearRegression#线性回归
# slr = LinearRegression()
# slr.fit(x_train, y_train)
# y_train_pred = slr.predict(x_train)
'''
[ 0.11231752  0.11384608  0.11419401  0.10853579  0.11640346]
MSE train: 806.422, test: 775.386
R^2 train: 0.119, test: 0.113
'''
#scores=cross_val_score(clf,x_test,y_test,cv=5)#[ 0.96434677  0.96455186  0.96257083  0.96886044  0.95664145]#MSE train: 0.000, test: 20.066#R^2 train: 1.000, test: 0.977
# from sklearn.ensemble import RandomForestRegressor#随机森林回归
# sample_leaf_options = [1,5,10,50,100,200,500]
# for leaf_size in sample_leaf_options :
# forest = RandomForestRegressor(n_estimators=1000, criterion='mse', oob_score = True,
#                                 max_features = "auto", min_samples_leaf = 50,random_state=1, n_jobs=-1)
# forest.fit(x_train, y_train)#MSE train: 21123.455, test: 842033.039
# from sklearn.ensemble import GradientBoostingRegressor
# gbdt=GradientBoostingRegressor(loss='ls',alpha=0.9
#                                ,n_estimators=500,
#                                learning_rate=0.05,
#                                max_depth=8,
#                                subsample=0.8,min_samples_split=9,max_leaf_nodes=10)
# from xgboost import XGBRegressor
# xgb=XGBRegressor(learning_rate=0.05, n_estimators=500)
scores=cross_val_score(clf,x_test,y_test,cv=5)
print(scores)
y_train_pred = clf.predict(x_train)
y_test_pred =clf.predict(x_test)
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))
plt.scatter(y_train_pred, y_train_pred - y_train, c='black', marker='o', s=35, alpha=0.5, label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', s=35, alpha=0.7, label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.tight_layout()
plt.savefig('./cnt_forest_residuals.png', dpi=300)
plt.show()
pre_tran_data_1=pd.read_csv('./pre_tran_train_1.csv')
pre_tran_data=pre_tran_data_1.drop(columns=['date_dt'])
pre_tran_data=pd.get_dummies(pre_tran_data,columns=['o_city_code','o_district_code','d_city_code','d_district_code','work_day'])
pre_tran_data=pre_tran_data.astype('float')
y_predict=clf.predict(pre_tran_data)
pre_tran_data_1['cnt']=y_predict
pre_tran_data_1.to_csv('./pre_tran_data.csv')




        