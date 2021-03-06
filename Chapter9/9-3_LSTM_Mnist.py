# _*_coding:utf-8_*_
"""
作   者：Charles Van
创建时间：2019/1/22 10:39
文   件：9-3_LSTM_Mnist.py
E-mail:willianan@hotmail.com
"""

import tensorflow as tf
#导入MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)

n_input = 28                                #MNIST data输入（img shape 28*28）
n_steps = 28                                #序列个数
n_hidden = 128                              #隐藏层个数
n_classes = 10                              #MNIST分类个数（0~9 digits）
#定义占位符
x = tf.placeholder("float",[None,n_steps,n_input])
y = tf.placeholder("float",[None,n_classes])

#BasicLSTMCell
x1 = tf.unstack(x,n_steps,1)
lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias = 1.0)
outputs,states = tf.contrib.rnn.static_rnn(lstm_cell,x1,dtype = tf.float32)

#2 LSTMCell
#lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0)
#outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x1, dtype=tf.float32)

#3 gru
#gru = tf.contrib.rnn.GRUCell(n_hidden)
#outputs = tf.contrib.rnn.static_rnn(gru, x1, dtype=tf.float32)

#4 创建动态RNN
#outputs,_  = tf.nn.dynamic_rnn(gru,x,dtype=tf.float32)
#outputs = tf.transpose(outputs, [1, 0, 2])

pred = tf.contrib.layers.fully_connected(outputs[-1],n_classes,activation_fn = None)

learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

#Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred,labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

#Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accurary = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#启动session
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	step = 1
	#keep training until reach max iterations
	while step * batch_size < training_iters:
		batch_x,batch_y = mnist.train.next_batch(batch_size)
		#Reshape data to get 28 seq of 28 elements
		batch_x = batch_x.reshape((batch_size,n_steps,n_input))
		#Run optimization op (backprop)
		sess.run(optimizer,feed_dict = {x:batch_x,y:batch_y})
		if step % display_step == 0:
			#计算批次数据的准确率
			acc = sess.run(accurary,feed_dict = {x:batch_x,y:batch_y})
			#Calculate batch loss
			loss = sess.run(cost,feed_dict = {x:batch_x,y:batch_y})
			print("Iter = " + str(step*batch_size) + ",Minibatch Loss = " + \
			      "{:,.6f}".format(loss) + ", Training Accuracy = " + \
			      "{:,.5f}".format(acc))
		step += 1
	print(" Finished!")
	
	#计算准确率 for 128 mnist test images
	test_len = 128
	test_data = mnist.test.images[:test_len].reshape((-1,n_steps,n_input))
	test_label = mnist.test.labels[:test_len]
	print("Testing Accuracy:",\
	      sess.run(accurary,feed_dict = {x:test_data,y:test_label}))