# _*_coding:utf-8_*_
"""
作   者：Charles Van
创建时间：2019/1/20 21:50
文   件：8-3_Pooling_Function.py
E-mail:willianan@hotmail.com
"""

import tensorflow as tf

img = tf.constant([[[0.0,4.0],[0.0,4.0],[0.0,4.0],[0.0,4.0]],
                   [[1.0,5.0],[1.0,5.0],[1.0,5.0],[1.0,5.0]],
                   [[2.0,6.0],[2.0,6.0],[2.0,6.0],[2.0,6.0]],
                   [[3.0,7.0], [3.0,7.0], [3.0,7.0], [3.0,7.0]]
                   ])
img = tf.reshape(img,[1,4,4,2])

pooling = tf.nn.max_pool(img,[1,2,2,1],[1,2,2,1],padding = 'VALID')
pooling1 = tf.nn.max_pool(img,[1,2,2,1],[1,1,1,1],padding = 'VALID')
pooling2 = tf.nn.avg_pool(img,[1,4,4,1],[1,1,1,1],padding = 'SAME')
pooling3 = tf.nn.avg_pool(img,[1,4,4,1],[1,4,4,1],padding = 'SAME')
nt_hpool2_flat = tf.reshape(tf.transpose(img),[-1,16])
pooling4 = tf.reduce_mean(nt_hpool2_flat,1)                               #1表示对行求均值，0表示对列求均值

with tf.Session() as sess:
	print("image:")
	image = sess.run(img)
	print(image)
	result = sess.run(pooling)
	print("result:\n",result)
	result = sess.run(pooling1)
	print("result1:\n",result)
	result = sess.run(pooling2)
	print("result2:\n",result)
	result = sess.run(pooling3)
	print("result3:\n",result)
	flat,result = sess.run([nt_hpool2_flat,pooling4])
	print("result4:\n",result)
	print("flat:\n",flat)