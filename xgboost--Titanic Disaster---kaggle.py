import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split,GridSearchCV,KFold
from sklearn.metrics import make_scorer,accuracy_score


train = pd.read_csv('Titanic Disaster/train.csv')
test = pd.read_csv('Titanic Disaster/test.csv')

del train['PassengerId']
del test['PassengerId']
del train['Name']
del test['Name']
del train['Ticket']
del test['Ticket']

#下面开始进行缺失值处理。对象有 cabin,embarked,age.
train['Cabin'] = train['Cabin'].isnull().astype(int)   #缺失值太多，按有无属性改为0/1
test['Cabin'] = test['Cabin'].isnull().astype(int)

#随机森林填补缺失值   ***DataFrame的处理是重点
data = train[['Survived','Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare', 'Cabin', 'Embarked']]
data.loc[data['Sex']=='male','Sex'] = 0
data.loc[data['Sex']=='female','Sex'] = 1
test.loc[test['Sex']=='male','Sex'] = 0
test.loc[test['Sex']=='female','Sex'] = 1
data.loc[data['Embarked']=='S','Embarked']=0
data.loc[data['Embarked']=='C','Embarked']=1
data.loc[data['Embarked']=='Q','Embarked']=2
test.loc[test['Embarked']=='S','Embarked']=0
test.loc[test['Embarked']=='C','Embarked']=1
test.loc[test['Embarked']=='Q','Embarked']=2
te = data.loc[data['Embarked'].notnull()]   #非空的embarked对应的行  .loc()先选行再选列，只有一个参数时默认为选行，列全取.
te_x = te.loc[:,['Survived','Pclass','Sex','SibSp', 'Parch','Fare', 'Cabin']].astype(float)
te_y = te.loc[:,['Embarked']].astype(float)
tr = data.loc[data['Embarked'].isnull()]  #缺失的embarked对应的行
tr_x = tr.loc[:,['Survived','Pclass', 'Sex','SibSp', 'Parch','Fare', 'Cabin']].astype(float)
fc = RandomForestClassifier()
fc.fit(te_x,te_y)
pr = fc.predict(tr_x)
data.loc[data['Embarked'].isnull(),'Embarked'] = pr

has_age = data.loc[data['Age'].notnull()]
not_has_age = data.loc[data['Age'].isnull()]
has_age_x = has_age.loc[:,['Survived','Pclass', 'Sex', 'SibSp', 'Parch','Fare', 'Cabin', 'Embarked']].astype(float)
has_age_y = has_age.loc[:,['Age']].astype(float)
not_has_age_x = not_has_age.loc[:,['Survived','Pclass', 'Sex', 'SibSp', 'Parch','Fare', 'Cabin', 'Embarked']].astype(float)
fr = RandomForestRegressor()
fr.fit(has_age_x,has_age_y)
pr_age = fr.predict(not_has_age_x)
data.loc[data['Age'].isnull(),'Age'] = pr_age

test_has_age = test.loc[test['Age'].notnull()]
test_not_has_age = test.loc[test['Age'].isnull()]
test_has_age_x = test_has_age.loc[:,['Pclass', 'Sex', 'SibSp', 'Parch','Fare', 'Cabin', 'Embarked']].astype(float)
test_has_age_y = test_has_age.loc[:,['Age']].astype(float)
test_not_has_age_x = test_not_has_age.loc[:,['Pclass', 'Sex', 'SibSp', 'Parch','Fare', 'Cabin', 'Embarked']].astype(float)
test_fr = RandomForestRegressor()
test_fr.fit(test_has_age_x,test_has_age_y)
test_pr_age = test_fr.predict(test_not_has_age_x)
test.loc[test['Age'].isnull(),'Age'] = test_pr_age

#将年龄划分为三个区间
data['child'] = 0
data.loc[data['Age'] <= 12,'child'] =1
data['adult'] = 0
data.loc[(data['Age']>12) & (data['Age']<=60),'adult']=1
data['olds'] = 0
data.loc[data['Age']>60,'olds'] =1
del data['Age']

test['child'] = 0
test.loc[test['Age'] <= 12,'child'] =1
test['adult'] = 0
test.loc[(test['Age']>12) & (test['Age']<=60),'adult']=1
test['olds'] = 0
test.loc[test['Age']>60,'olds'] =1
del test['Age']

#归一化 Fare
data['Fare'] = (data['Fare']-data['Fare'].mean())/(data['Fare'].max()-data['Fare'].min())
test['Fare'] = (test['Fare']-test['Fare'].mean())/(test['Fare'].max()-test['Fare'].min())


#对 pclass,sex,cabin,embarked 进行one-hot编码
data['pclass_0'] = 0
data['pclass_1'] = 0
data['pclass_2'] = 0
data.loc[data['Pclass']==1,'pclass_0'] = 1
data.loc[data['Pclass']==2,'pclass_1'] = 1
data.loc[data['Pclass']==3,'pclass_2'] = 1
test['pclass_0'] = 0
test['pclass_1'] = 0
test['pclass_2'] = 0
test.loc[test['Pclass']==1,'pclass_0'] = 1
test.loc[test['Pclass']==2,'pclass_1'] = 1
test.loc[test['Pclass']==3,'pclass_2'] = 1

data['sex_0'] = 0
data['sex_1'] = 0
data.loc[data['Sex']==0,'sex_0'] = 1
data.loc[data['Sex']==1,'sex_1'] = 1
test['sex_0'] = 0
test['sex_1'] = 0
test.loc[test['Sex']==0,'sex_0'] = 1
test.loc[test['Sex']==1,'sex_1'] = 1

data['cabin_0'] = 0
data['cabin_1'] = 0
data.loc[data['Cabin']==0,'cabin_0'] = 1
data.loc[data['Cabin']==1,'cabin_1'] = 1
test['cabin_0'] = 0
test['cabin_1'] = 0
test.loc[test['Cabin']==0,'cabin_0'] = 1
test.loc[test['Cabin']==1,'cabin_1'] = 1

data['embarked_0'] = 0
data['embarked_1'] = 0
data['embarked_2'] = 0
data.loc[data['Embarked']==0,'embarked_0'] = 1
data.loc[data['Embarked']==1,'embarked_1'] = 1
data.loc[data['Embarked']==2,'embarked_2'] = 1
test['embarked_0'] = 0
test['embarked_1'] = 0
test['embarked_2'] = 0
test.loc[test['Embarked']==0,'embarked_0'] = 1
test.loc[test['Embarked']==1,'embarked_1'] = 1
test.loc[test['Embarked']==2,'embarked_2'] = 1

del data['Pclass']
del data['Sex']
del data['Cabin']
del data['Embarked']
del test['Pclass']
del test['Sex']
del test['Cabin']
del test['Embarked']

#一等舱的女人、小孩
data['class1fc'] = 0
data.loc[(data['pclass_0']==1) & (((data['sex_1']==1) & (data['adult']==1))|(data['child']==1)),'class1fc'] = 1
test['class1fc'] = 0
test.loc[(test['pclass_0']==1) & (((test['sex_1']==1) & (test['adult']==1))|(test['child']==1)),'class1fc'] = 1

data['class2fc'] = 0
data.loc[(data['pclass_1']==1) & (((data['sex_1']==1) & (data['adult']==1))|(data['child']==1)),'class2fc'] = 1
test['class2fc'] = 0
test.loc[(test['pclass_1']==1) & (((test['sex_1']==1) & (test['adult']==1))|(test['child']==1)),'class2fc'] = 1

data['class3fc'] = 0
data.loc[(data['pclass_2']==1) & (((data['sex_1']==1) & (data['adult']==1))|(data['child']==1)),'class3fc'] = 1
test['class3fc'] = 0
test.loc[(test['pclass_2']==1) & (((test['sex_1']==1) & (test['adult']==1))|(test['child']==1)),'class3fc'] = 1



"""
#母亲
data['mother'] = 0
data.loc[(data['sex_1']==1)&(data['adult']==1)&(data['Parch']!=0),'mother'] = 1
test['mother'] = 0
test.loc[(test['sex_1']==1)&(test['adult']==1)&(test['Parch']!=0),'mother'] = 1
"""
#家庭大小
data['family_num'] = data['SibSp']+data['Parch']
del data['SibSp']
del data['Parch']

test['family_num'] = test['SibSp']+test['Parch']
del test['SibSp']
del test['Parch']

data['family_0']= 0
data.loc[(data['family_num']>=2) & (data['family_num']<=4),'family_0'] = 1
data['family_1'] =0
data.loc[(data['family_num']==1) | ((data['family_num']<=8) & (data['family_num']>4)),'family_1'] = 1
data['family_2'] = 0
data.loc[data['family_num']>8,'family_2'] = 1
del data['family_num']

test['family_0']= 0
test.loc[(test['family_num']>=2) & (test['family_num']<=4),'family_0'] = 1
test['family_1'] =0
test.loc[(test['family_num']==1) | ((test['family_num']<=8) & (test['family_num']>4)),'family_1'] = 1
test['family_2'] = 0
test.loc[test['family_num']>8,'family_2'] = 1
del test['family_num']


x = data.iloc[:,1:].values
y = data['Survived'].values
test = test.values

x_train,x_valid,y_train,y_valid = train_test_split(x,y,train_size=0.8)
xgbtrain_valid = xgb.DMatrix(x_train,y_train)
xgbtest_valid = xgb.DMatrix(x_valid)

xgbtrain = xgb.DMatrix(x,y)
xgbtest = xgb.DMatrix(test)

"""
#网格搜索法
classifier = xgb.XGBClassifier(random_state=0)
params = {'max_depth':[4,5,6],'gamma':[0.04,0.05,0.06]}
scoring = make_scorer(accuracy_score)

grid = GridSearchCV(classifier,params,scoring,cv=10)
grid = grid.fit(x,y)
clf = grid.best_estimator_
print('best score:%f'%grid.best_score_)
print('best params:',grid.best_params_)
print('best estimator:',clf)
print('split_test score: %f'%clf.score(x,y))
"""


params = {
        'booster':'gbtree',
        'base_score':0.5,
        'eta':0.3,
        'gamma':0.05,
        'max_depth':6,
        'subsample':1,
        'colsample_bytree':1,
        'objective':'binary:logistic',
        'eval_metric': 'auc',
        'seed':0,
        'learning_rates':0.1,
        'max_delta_step':0,
        'min_child_weight':1, 
        'n_estimators':100,
        'n_jobs':1, 
        'silent':True
        }


xgbmodel = xgb.train(params = params,dtrain=xgbtrain_valid)
pre_valid = xgbmodel.predict(xgbtest_valid).tolist()
for i in range(len(pre_valid)):
    if pre_valid[i] >= 0.5:
        pre_valid[i] = 1
    else:
        pre_valid[i] = 0

acc_valid = (np.array(pre_valid) == y_valid).sum()/len(y_valid)
print('acc_valid:',acc_valid)  #0.8491



#xgbmodel = xgb.train(params=params,dtrain=xgbtrain)
pre4 = xgbmodel.predict(xgbtest).tolist()
for i in range(len(pre4)):
    if pre4[i] >= 0.5:
        pre4[i] = 1
    else:
        pre4[i] = 0

sub4=pd.DataFrame({"PassengerId": list(range(892,892+len(pre4))),"Survived": pre4})
sub4.to_csv('Titanic Disaster/sub4.csv', index=False, header=True)


#0.77033