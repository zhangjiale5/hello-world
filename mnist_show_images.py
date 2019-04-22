"""
show mnist_images
"""


import numpy as np
import lenet5_backward
from PIL import Image
import os

mnist = lenet5_backward.mnist

test_images = mnist.test.images
test_images2 = np.reshape(test_images,(10000,28,28))
test_images3 = 255*test_images2  
#将np数组转化为图片保存， 用到Image模块
test_images4 = np.asanyarray(test_images3,dtype=np.uint8) #dtype 一定要写 ，不然图片生成不对
for i in range(10000):
    test_images5 = Image.fromarray(test_images4[i],'L')
    test_images5.save(os.path.join('./mnist_images/','%d.png'%i))

