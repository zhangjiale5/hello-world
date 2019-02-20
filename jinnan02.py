# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 16:34:36 2019

@author: 张家乐
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold,RepeatedKFold
import re
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss

train = pd.read_csv('E:\desktop\jinnan_round1_train_20181227.csv')
test = pd.read_csv('E:\desktop\jinnan_round1_testA_20181227.csv')
train.drop(index=[120,869,484,410,687,762,1195],inplace=True)

#时刻：A5,A7,A9,A11,A14,A16,A24,A26,B5,B7.   时间：A20,A28,B4,B9,B10,B11.

# 合并数据集
target = train['收率']
del train['收率']
data = pd.concat([train,test],axis=0,ignore_index=True)
data = data.loc[:,['A2','A3','A5','A6','A7','A8','A9','A10','A11','A12','A14','A15','A16','A17','A19',
                   'A20','A21','A22','A24','A25','A26','A27','A28','B1','B2','B4','B5','B6','B7','B8',
                   'B9','B10','B11','B12','B14']]
data = data.fillna(-1)

# 日期中有些输入错误和遗漏
def t2s(t):
    try:
        t,m,s=t.split(":")
    except:
        if t=='1900/1/9 7:00':
            return np.nan
        elif t=='1900/1/1 2:30':
            return np.nan
        elif t==-1:
            return np.nan
        else:
            return np.nan
    try:
        tm = int(t)*3600+int(m)*60+int(s)
    except:
        return np.nan
    return tm
for f in ['A5','A7','A9','A11','A14','A16','A24','A26','B5','B7']:
    data[f] = data[f].apply(t2s)/3600

def getDuration(se):
    try:
        sh,sm,eh,em=re.split("[:,-]",se)
    except:
        if se=='14::30-15:30':
            return 3600
        elif se=='13；00-14:00':
            return 3600
        elif se=='21:00-22；00':
            return 3600
        elif se=='22"00-0:00':
            return 7200
        elif se=='2:00-3;00':
            return 3600
        elif se=='1:30-3;00':
            return 5400
        elif se=='15:00-1600':
            return 3600
        elif se==-1:
            return np.nan
        else:
            return np.nan  
    try:
        tm = int(eh)*3600+int(em)*60-int(sm)*60-int(sh)*3600
        if tm <-1:
            tm=tm+24*3600
        else:
            pass
    except:
        if se=='19:-20:05':
            return 3600
        else:
            return np.nan
    return tm
for f in ['A20','A28','B4','B9','B10','B11']:
    data[f] = data.apply(lambda df: getDuration(df[f]), axis=1)/3600
"""
#label encoder
    cate_columns = [f for f in data.columns]
for f in cate_columns:
    data[f] = data[f].map(dict(zip(data[f].unique(), range(0, data[f].nunique()))))

"""
#A2与A3缺失值互补
data.A2=data.A2.fillna(0)
data.A3=data.A3.fillna(0)
data.A3=data.A2+data.A3
data.drop('A2',axis=1,inplace=True)

data.B1=data.B1.fillna(0)
data.B2=data.B2.fillna(0)
data.B1 = data.B1*data.B2
data.B1[data.B1==0]=320*3.5
data.drop('B2',axis=1,inplace=True)

train = data[:train.shape[0]]
test  = data[train.shape[0]:]

cate_columns = [f for f in data.columns]

train['target'] = list(target) 
train['intTarget'] = pd.cut(train['target'], 5, labels=False) 
train = pd.get_dummies(train, columns=['intTarget']) 
li = train.columns[-5:] 
mean_features = []
for f1 in cate_columns:
    rate = train[f1].value_counts(normalize=True, dropna=False).values[0]
    if rate < 0.50:
        for f2 in li:
            col_name = f1+"_"+f2+'_mean'
            mean_features.append(col_name)
            order_label = train.groupby([f1])[f2].mean()
            for df in [train, test]:
                df[col_name] = df[f].map(order_label)
train.drop(li, axis=1, inplace=True)

train.drop(['target'], axis=1, inplace=True)
X_train = train.values
y_train = target.values
X_test = test.values

param = {'num_leaves': 120,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.01,
         "min_child_samples": 30,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'mse',
         "lambda_l1": 0.1,
         "verbosity": -1}
folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_lgb = np.zeros(len(train))
predictions_lgb = np.zeros(len(test))


for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_+1))
    trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
    val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 100)
    oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
    
    predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, target)))


##### xgb
xgb_params = {'eta': 0.005, 'max_depth': 10, 'subsample': 0.8, 'colsample_bytree': 0.8, 
          'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 4}

folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_xgb = np.zeros(len(train))
predictions_xgb = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_+1))
    trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
    val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
    clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=100, params=xgb_params)
    oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
    predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits
    
print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, target)))



# 将lgb和xgb的结果进行stacking
train_stack = np.vstack([oof_lgb,oof_xgb]).transpose()
test_stack = np.vstack([predictions_lgb, predictions_xgb]).transpose()

folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4590)
oof_stack = np.zeros(train_stack.shape[0])
predictions = np.zeros(test_stack.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack,target)):
    print("fold {}".format(fold_)) 
    trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values
    
    clf_3 = BayesianRidge()
    clf_3.fit(trn_data, trn_y)
    
    oof_stack[val_idx] = clf_3.predict(val_data)
    predictions += clf_3.predict(test_stack) / 10
    
print(mean_squared_error(target.values, oof_stack))


sub_df = pd.read_csv('E:\desktop\jinnan_round1_submit_20181227.csv', header=None)
sub_df[1] = predictions
sub_df[1] = sub_df[1].apply(lambda x:round(x, 3))
sub_df.to_csv("jinnan02.csv", index=False, header=None)


