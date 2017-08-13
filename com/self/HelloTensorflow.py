# -*- coding: utf-8 -*- 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.random.rand(100).astype(np.float32)
# noise = np.random.uniform(0.01, 0.03, 1)
y_data = x_data * 0.1 + 0.3

plt.scatter(x_data, y_data)
plt.show()

###############
Weights = tf.Variable(tf.random_uniform([1], -1, 1, tf.float32))
biases = tf.Variable(tf.zeros([1], tf.float32))

##################
y = x_data * Weights + biases
loss = tf.reduce_mean(tf.square(y_data - y))
#################
train_op = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)

plt.ion()

for i in range(101):
    sess.run(train_op)
    if i % 2 == 0:
        plt.cla()
        plt.scatter(x_data, y_data)
        plt.plot(x_data, sess.run(y), 'r-', lw=5)
    #         plt.text(0.5, 0, 'Loss=%.4f' % sess.run(loss))
        plt.pause(0.1)
    #         print(i, sess.run(Weights), sess.run(biases))

plt.ioff()
plt.show()


