import tensorflow as tf
global_step = tf.Variable(0,trainable=False)
initial_learing_rate = 0.1                                  #初始学习率
learning_rate = tf.train.exponential_decay(initial_learing_rate,global_step=global_step,decay_steps=10,decay_rate=0.9)
opt = tf.train.GradientDescentOptimizer(learning_rate)
add_global = global_step.assign_add(1)                      #定义一个op，令global_step加1完成记步
with tf.Session() as sess:
	tf.global_variables_initializer().run()
	print(sess.run(learning_rate))
	for i in range(20):
		g,rate = sess.run([add_global,learning_rate])       #循环20步，将每步的学习率打印出来
		print(g,rate)