from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import tensorflow as tf
savedir = "log/"
print_tensors_in_checkpoint_file(savedir + "linermodel.cpkt",None,True)

W = tf.Variable(1.0,name = "weight")
b = tf.Variable(2.0,name = "bias")
saver = tf.train.Saver({'weight':b,'bias':W})
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver.save(sess,savedir + "linermodel.cpkt")
print_tensors_in_checkpoint_file(savedir + "linermodel.cpkt",None,True)