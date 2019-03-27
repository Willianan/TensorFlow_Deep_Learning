# _*_coding:utf-8_*_
"""
作   者：Charles Van
创建时间：2019/1/22 16:42
文   件：9-18_Multi_Two-way_RNN.py
E-mail:willianan@hotmail.com
"""

import tensorflow as tf
from tensorflow.contrib import rnn
# 导入 MINST 数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets( "MNIST_data/", one_hot = True )

# 参数设置
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 28  # MNIST data 输入 (img shape: 28*28)
n_steps = 28  # timesteps
n_hidden = 128  # hidden layer num of features
n_classes = 10  # MNIST 列别 (0-9 ，一共10类)

tf.reset_default_graph( )

# tf Graph input
x = tf.placeholder( "float", [ None, n_steps, n_input ] )
y = tf.placeholder( "float", [ None, n_classes ] )

x1 = tf.unstack( x, n_steps, 1 )
lstm_fw_cell = rnn.BasicLSTMCell( n_hidden, forget_bias = 1.0 )
lstm_bw_cell = rnn.BasicLSTMCell( n_hidden, forget_bias = 1.0 )

stacked_rnn = [ ]
stacked_bw_rnn = [ ]
for i in range( 3 ):
	stacked_rnn.append( tf.contrib.rnn.LSTMCell( n_hidden ) )
	stacked_bw_rnn.append( tf.contrib.rnn.LSTMCell( n_hidden ) )

mcell = tf.contrib.rnn.MultiRNNCell( stacked_rnn )
mcell_bw = tf.contrib.rnn.MultiRNNCell( stacked_bw_rnn )

outputs, _, _ = rnn.stack_bidirectional_rnn( [ mcell ], [ mcell_bw ], x1, dtype = tf.float32 )

pred = tf.contrib.layers.fully_connected( outputs[ -1 ], n_classes, activation_fn = None )

# Define loss and optimizer
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( logits = pred, labels = y ) )
optimizer = tf.train.AdamOptimizer( learning_rate = learning_rate ).minimize( cost )

# Evaluate model
correct_pred = tf.equal( tf.argmax( pred, 1 ), tf.argmax( y, 1 ) )
accuracy = tf.reduce_mean( tf.cast( correct_pred, tf.float32 ) )

# 启动session
with tf.Session( ) as sess:
	sess.run( tf.global_variables_initializer( ) )
	step = 1
	# Keep training until reach max iterations
	while step * batch_size < training_iters:
		batch_x, batch_y = mnist.train.next_batch( batch_size )
		# Reshape data to get 28 seq of 28 elements
		batch_x = batch_x.reshape( (batch_size, n_steps, n_input) )
		# Run optimization op (backprop)
		sess.run( optimizer, feed_dict = { x: batch_x, y: batch_y } )
		if step % display_step == 0:
			# 计算批次数据的准确率
			acc = sess.run( accuracy, feed_dict = { x: batch_x, y: batch_y } )
			# Calculate batch loss
			loss = sess.run( cost, feed_dict = { x: batch_x, y: batch_y } )
			print( "Iter " + str( step * batch_size ) + ", Minibatch Loss = " + \
			       "{:.6f}".format( loss ) + ", Training Accuracy = " + \
			       "{:.5f}".format( acc ) )
		step += 1
	print( " Finished!" )
	
	# 计算准确率 for 128 mnist test images
	test_len = 128
	test_data = mnist.test.images[ :test_len ].reshape( (-1, n_steps, n_input) )
	test_label = mnist.test.labels[ :test_len ]
	print( "Testing Accuracy:", \
	       sess.run( accuracy, feed_dict = { x: test_data, y: test_label } ) )