# -*- coding: utf-8 -*-
'''
Created on 2018年12月7日

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
'''
生成训练集

a={}
j=1
flow_train_1=pd.read_csv('./flow_train_1.csv')
for i in flow_train_1['date_dt'].unique():
    a[i]=j
    j+=1
series_num=[]
for i in flow_train_1['date_dt']:
    series_num.append(a[i])
flow_train_1['series_num']=series_num
flow_train_1=flow_train_1.drop(columns=['date_dt'])
flow_train_1=pd.get_dummies(flow_train_1,columns=['work_day','city_code','district_code'])
flow_train_1.to_csv('./train_flow.csv')
'''
dataset=pd.read_csv('./train_flow.csv')
dataset=dataset.astype('float')
label=dataset['dwell']
feature_set=dataset.drop(columns=['dwell'])
x_train,x_test,y_train,y_test=train_test_split(feature_set,label,random_state=1)
# clf=tree.DecisionTreeRegressor()#决策树回归模型
# clf=clf.fit(x_train,y_train)
'''
[ 0.98711002  0.99093382  0.98967206  0.98792599  0.99269528]
MSE train: 0.000, test: 1109.878
R^2 train: 1.000, test: 0.993
'''
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
[ 0.98641737  0.98783323  0.98564563  0.98507105  0.98869409]
MSE train: 1986.068, test: 1981.567
R^2 train: 0.987, test: 0.987
'''
#scores=cross_val_score(clf,x_test,y_test,cv=5)#[ 0.96434677  0.96455186  0.96257083  0.96886044  0.95664145]#MSE train: 0.000, test: 20.066#R^2 train: 1.000, test: 0.977
# from sklearn.ensemble import RandomForestRegressor#随机森林回归
# sample_leaf_options = [1,5,10,50,100,200,500]
# for leaf_size in sample_leaf_options :
# forest = RandomForestRegressor(n_estimators=1000, criterion='mse', oob_score = True,
#                                 max_features = "auto", min_samples_leaf = 50,random_state=1, n_jobs=-1)
# forest.fit(x_train, y_train)
'''
[ 0.91395514  0.91200402  0.91785267  0.90933813  0.909192  ]
MSE train: 2392.672, test: 2481.888
R^2 train: 0.984, test: 0.984
'''

from sklearn.ensemble import GradientBoostingRegressor
gbdt=GradientBoostingRegressor(loss='ls',alpha=0.9
                               ,n_estimators=500,
                               learning_rate=0.05,
                               max_depth=8,
                               subsample=0.8,min_samples_split=9,max_leaf_nodes=10)
'''
[ 0.99274817  0.99349717  0.99408877  0.99240158  0.99441201]
MSE train: 598.379, test: 826.408
R^2 train: 0.996, test: 0.995
'''

# from xgboost import XGBRegressor
# xgb=XGBRegressor(learning_rate=0.05, n_estimators=500)
'''
[ 0.98967751  0.99087634  0.99171368  0.98947243  0.99166763]
MSE train: 1189.055, test: 1264.184
R^2 train: 0.992, test: 0.992
'''
gbdt.fit(x_train, y_train)
scores=cross_val_score(gbdt,x_test,y_test,cv=5)
print(scores)
y_train_pred = gbdt.predict(x_train)
y_test_pred =gbdt.predict(x_test)
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
plt.savefig('./dwell_gbdt_residuals.png', dpi=300)
pre_tran_data_1=pd.read_csv('./pre_flow_1.csv')
pre_tran_data=pre_tran_data_1.drop(columns=['date_dt'])
pre_tran_data=pd.get_dummies(pre_tran_data,columns=['work_day','city_code','district_code'])
pre_tran_data=pre_tran_data.astype('float')
y_predict=gbdt.predict(pre_tran_data)
pre_tran_data_1['dwell']=y_predict
pre_tran_data_1.to_csv('./pre_flow_gbdt_data.csv')   