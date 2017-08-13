'''
Created on Aug 12, 2017

@author: hadoop
'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np



def add_layer(inputs, input_size, output_size, activation_function = None):
    Weights = tf.Variable(tf.random_normal([input_size, output_size]))
    biases = tf.Variable(tf.zeros([1, output_size])) + 0.1
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function == None:
        outputs =  Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#########
# x_data = np.random.uniform(-1, 1, 3)[:, np.newaxis]
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

# plt.scatter(x_data, y_data)
# plt.show()
xs = tf.placeholder(tf.float32, x_data.shape)
ys = tf.placeholder(tf.float32, y_data.shape)

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

plt.ion()
for i in range(1001):
    _, l, pred = sess.run([train_op, loss, prediction], {xs: x_data, ys: y_data})
     
    if i % 20 == 0:
        plt.cla()
        plt.scatter(x_data, y_data)
        plt.plot(x_data, pred, 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show() 