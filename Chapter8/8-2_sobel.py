# _*_coding:utf-8_*_
"""
作   者：Charles Van
创建时间：2019/1/20 21:18
文   件：8-2_sobel.py
E-mail:willianan@hotmail.com
"""

#sobel算子
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

myimg = mpimg.imread('img.png')                     #读取和代码处于同一目录下的图片
plt.imshow(myimg)                                   #显示图片
plt.axis('off')                                     #不显示坐标轴
plt.show()
print(mpimg.shape)

full = np.reshape(myimg,[13264,2448,3])
inputfull = tf.Variable(tf.constant(1.0,shape = [1,3264,2448,3]))
filter = tf.Variable(tf.constant([[-1.0,-1.0,-1.0],[0,0,0],[1.0,1.0,1.0],
                                 [-2.0,-2.0,-2.0],[0,0,0],[2.0,2.0,2.0],
                                 [-1.0,-1.0,-1.0],[0,0,0],[1.0,1.0,1.0]],shape = [3,3,3,1]))
op = tf.nn.conv2d(inputfull,filter,strides = [1,1,1,1],padding = 'SAME')            #3个通道输入，生成一个feature ma
o = tf.cast((op - tf.reduce_min(op) / tf.reduce_max(op) - tf.reduce_min(op))*255,tf.uint8)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	t,f = sess.run([o,filter],feed_dict = {inputfull:full})
	t = np.reshape(t,[3264,2448])
	plt.imshow(t,cmap = 'Greys_r')
	plt.axis('off')
	plt.show()