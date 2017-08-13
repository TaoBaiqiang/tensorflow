'''
Created on Aug 12, 2017

@author: hadoop
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(-1, 1, 100, np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.1, size=x_data.shape)
y_data = np.power(x_data, 2) + noise

# plt.scatter(x_data, y_data)
# plt.show()

xs = tf.placeholder(tf.float32, x_data.shape)
ys = tf.placeholder(tf.float32, y_data.shape)

l1 = tf.layers.dense(xs, 10, tf.nn.relu)
prediction = tf.layers.dense(l1, 1)

loss = tf.losses.mean_squared_error(ys, prediction)
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion()

for i in range(101):
    _, l, pred = sess.run([train_step, loss, prediction], {xs: x_data, ys: y_data})
    if i % 20 == 0:
        plt.cla()
        plt.scatter(x_data, y_data)
        plt.plot(x_data, pred, 'r-', lw=3)
        plt.text(0.5, 0, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()