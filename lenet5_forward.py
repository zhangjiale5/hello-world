"""
LeNet5_forward_model
"""



import tensorflow as tf

def get_weight(shape,regularizer):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w
def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#定义前向传播网络
def forward(x,train,regularizer):
    conv1_w = get_weight([5,5,1,32],regularizer)
    conv1_b = get_bias([32])
    conv1 = conv2d(x,conv1_w)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_b))
    pool1 = max_pool_2x2(relu1)
    
    conv2_w = get_weight([5,5,32,64],regularizer)
    conv2_b = get_bias([64])
    conv2 = conv2d(pool1,conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_b))
    pool2 = max_pool_2x2(relu2)
    
    pool_shape = pool2.get_shape().as_list()  #[batch_size,7,7,64]
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]  #7*7*64
    reshaped = tf.reshape(pool2,[pool_shape[0],nodes])
    
    fc1_w = get_weight([nodes,512],regularizer)
    fc1_b = get_bias([512])
    fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_w)+fc1_b)
    
    if train:
        fc1 = tf.nn.dropout(fc1,0.5)
    
    fc2_w = get_weight([512,10],regularizer)
    fc2_b = get_bias([10])
    y = tf.matmul(fc1,fc2_w)+fc2_b
    return y

