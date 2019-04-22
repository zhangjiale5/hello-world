"""
LeNet5_test
"""



import tensorflow as tf
import numpy as np
import lenet5_forward
import lenet5_backward
import time

mnist = lenet5_backward.mnist
test_interval_secs = 5

def test(mnist):   
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,[mnist.test.num_examples,28,28,1])
        y_= tf.placeholder(tf.float32,[None,10])
        y = lenet5_forward.forward(x,False,None)
        
        #加载模型中参数的滑动平均值L
        ema = tf.train.ExponentialMovingAverage(lenet5_backward.moving_average_decay)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore) #实例化saver对象，实现参数滑动平均值的加载
        
        pred = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(pred,tf.float32))
        
        while True:
            #在测试网络效果时，需要将训练好的伸进网络模型加载
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(lenet5_backward.model_save_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    
                    xs = np.reshape(mnist.test.images,(mnist.test.num_examples,28,28,1))
                    accuracy_score = sess.run(accuracy,feed_dict={x:xs,y_:mnist.test.labels})
                    print('After %s training steps,test accuracy is %g'%(global_step,accuracy_score))
                else:
                    print('No checkpoint file found')
            time.sleep(test_interval_secs)
                
def main():
    test(mnist)

if __name__ == '__main__':
    main()
    