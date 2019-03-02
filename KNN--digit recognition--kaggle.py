import pandas as pd
import numpy as np
import time

train = pd.read_csv('digit recognition/train.csv')
test = pd.read_csv('digit recognition/test.csv')

train_label = train.iloc[:,0].values.astype('float32')  
""" DataFrame.values 提取值, type(train_label)=numpy.ndarray"""
train_data = train.iloc[:,1:].values.astype('float32')
test_data = test.values.astype('float32')

#定义分类每个样本的函数
def classify(inX,train_data,train_label,k):
    inX = np.mat(inX)
    train_data = np.mat(train_data)
    train_label = np.mat(train_label)
    datasetsize = train_data.shape[0]
    diffmat = np.tile(inX,(datasetsize,1)) - train_data 
#将一个一维行向量扩展后 与 训练数据矩阵做减法    
    sq_diffmat = np.array(diffmat)**2
    sq_distances = sq_diffmat.sum(axis=1)
    distances = sq_distances**0.5
    sorted_distance_index = distances.argsort()  
#argsort() 将数组从小到大排列后,返回其索引   
    classcount={}
    for i in range(k):
        votelabel = train_label[0,sorted_distance_index[i]]   #为什么需要“0”？？？
        classcount[votelabel] = classcount.get(votelabel,0)+1
#dict.get(key,default=None)
    pred_label = sorted(zip(classcount.values(),classcount.keys()))[-1][1]
    return pred_label

def main():
    testsize = test_data.shape[0]
    pred = []
    for i in range(testsize):
        pred_label = classify(test_data[i],train_data,train_label,5)
        pred.append(pred_label)
    return pred

start = time.time()    
pred = main()
end = time.time()

#输出预测数据
sub=pd.DataFrame({"ImageId": list(range(1,len(pred)+1)),"Label": pred})
sub.to_csv('digit recognition/sub.csv', index=False, header=True)
print('finished')
print('cost time: %f mins'%(end-start)/60)
