"""
LeNet5_backward_model
"""


import tensorflow as tf
import numpy as np
import os
import lenet5_forward
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./data/',one_hot=True)

regularizer = 0.0001 #正则化项系数
batch_size = 100
learning_rate_base = 0.005 #初始学习率
learning_rate_decay = 0.99
moving_average_decay = 0.99
model_save_path = './lenet5_model/'
model_name = 'lenet5_model'
steps = 10000


def backward(mnist):
    x = tf.placeholder(tf.float32,[batch_size,28,28,1])
    y_=tf.placeholder(tf.float32,[None,10])
    y = lenet5_forward.forward(x,True,regularizer)
    global_step = tf.Variable(0,trainable=False)
    
    #交叉熵
    ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1)))
    loss = ce + tf.add_n(tf.get_collection('losses'))
    
    #指数衰减学习率 decayed_learning_rate=learning_rate*decay_rate^(global_step/decay_steps)
    learning_rate = tf.train.exponential_decay(
            learning_rate_base,
            global_step,
            mnist.train.num_examples/batch_size,
            learning_rate_decay,
            staircase=True)
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    
    #使用滑动平均的方法更新参数， shadow_variable = decay*shadow_variable+(1-ddcay)*variable
    ema = tf.train.ExponentialMovingAverage(moving_average_decay,global_step) #decay越大模型越稳定
    ema_op = ema.apply(tf.trainable_variables())
    
    #将train_step和 ema_op两个训练操作绑定到train_op 上
    with tf.control_dependencies([train_step,ema_op]):
        train_op = tf.no_op(name='train')  #tf.no_op什么也不做
    
    saver = tf.train.Saver() #实例化saver对象
    
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        ckpt = tf.train.get_checkpoint_state(model_save_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
            
        for i in range(steps):
            xs,ys=mnist.train.next_batch(batch_size)
            xs = np.reshape(xs,[batch_size,28,28,1])
            _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            if i % 100 == 0:
                print('After %d training steps,loss on training batch is %g'%(step,loss_value))
                saver.save(sess,os.path.join(model_save_path,model_name),global_step=global_step)
                
def main():
    backward(mnist)
    
if __name__ == '__main__':
    main()
            