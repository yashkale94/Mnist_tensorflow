import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import fully_connected

mnist=input_data.read_data_sets("/MNIST_data/",one_hot=True)

num_inputs = 784

num_neurons_hid1 = 500
num_neurons_hid2 = 250

num_neurons_hid3 = 500
num_output = 784

learning_rate = 0.01

activation_function = tf.nn.relu

X = tf.placeholder(tf.float32, shape =  [None,num_inputs])
initializer = tf.variance_scaling_initializer()

weights_1 = tf.Variable(initializer([num_inputs, num_neurons_hid1]),dtype=tf.float32)
weights_2 = tf.Variable(initializer([num_neurons_hid1, num_neurons_hid2]),dtype=tf.float32)
weights_3 = tf.Variable(initializer([num_neurons_hid2, num_neurons_hid3]),dtype=tf.float32)
weights_4 = tf.Variable(initializer([num_neurons_hid3, num_output]),dtype=tf.float32)

bias_1 = tf.Variable(tf.zeros([num_neurons_hid1]), dtype = tf.float32)
bias_2 = tf.Variable(tf.zeros([num_neurons_hid2]), dtype = tf.float32)
bias_3 = tf.Variable(tf.zeros([num_neurons_hid3]), dtype = tf.float32)
bias_4 = tf.Variable(tf.zeros([num_output]), dtype = tf.float32)

act1 = activation_function(tf.matmul(X,weights_1)+bias_1)
act2 = activation_function(tf.matmul(act1,weights_2)+bias_2)
act3 = activation_function(tf.matmul(act2,weights_3)+bias_3)
act4 = activation_function(tf.matmul(act3,weights_4)+bias_4)

loss = tf.reduce_mean(tf.square(act4 - X))

optimizer = tf.train.AdamOptimizer(learning_rate)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

num_epochs = 2
batch_size = 200

num_test = 10
print('HI')

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init)
	print('HI2')
	for epoch in range(num_epochs):
		num_batches = mnist.train.num_examples#//batch_size
		for iteration in range(num_batches):
			X_batch,y_batch = mnist.train.next_batch(batch_size)
			sess.run(train,feed_dict={X:X_batch})


		train_loss = loss.eval(feed_dict={X:X_batch})
		print("epoch {} loss {}".format(epoch,train_loss))

	saver.save(sess, 'model_final')

# with tf.Session() as sess:  
# #First let's load meta graph and restore weights
# 	sess.run(init)
# 	saver = tf.train.import_meta_graph('model_final.meta')
# 	saver.restore(sess,tf.train.latest_checkpoint('./'))

	results = act4.eval(feed_dict={X:mnist.test.images[:num_test]})
    
    #Comparing original images with reconstructions
	f,a=plt.subplots(2,10,figsize=(20,4))
	for i in range(num_test):
		a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
		a[1][i].imshow(np.reshape(results[i],(28,28)))
	
	plt.show()


