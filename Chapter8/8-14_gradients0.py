# _*_coding:utf-8_*_
"""
作   者：Charles Van
创建时间：2019/1/21 17:12
文   件：8-14_gradients0.py
E-mail:willianan@hotmail.com
"""

import tensorflow as tf

w1 = tf.Variable([[1,2]])
w2 = tf.Variable([[3,4]])

y = tf.matmul(w1,[[9],[10]])
grads = tf.gradients(y,[w1])

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	gradval = sess.run(grads)
	print(gradval)