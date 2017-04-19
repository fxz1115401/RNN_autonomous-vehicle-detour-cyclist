'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
print(mnist)
'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

import csv
data_with_cyc = [];
data_with_cyc_str = [];
batch_size = 20;

training = 0
with open('cyclistWith3sAfter.csv', 'rb') as csvfile:
    data_pre = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in data_pre:
        if row[4] != 'NULL':
            del row[26:36]
            del row[17:24]
            del row[9:14]
            del row[1:6]
            data_with_cyc_str.append(row)

del data_with_cyc_str[0]

for row in data_with_cyc_str:
    float_row = []
    for col in row:

        col = col.strip()
        float_row.append( float(col))
    data_with_cyc.append(float_row)

data_with_cyc_after_temp = []
data_with_cyc_after_output_temp = []

for i in range(len(data_with_cyc) - 9):
    if  data_with_cyc[i][0] == data_with_cyc[i + 9][0]:
        for j in range(10):
            data_with_cyc_after_output_temp.append(list(data_with_cyc[i+j][-2:]))
            temp = list(data_with_cyc[i + j])
            del temp[0]
            del temp[-2:]
            data_with_cyc_after_temp.append(temp)
#print data_no_cyc_after



data_with_cyc_after = [];
data_with_cyc_after_output = [];
for i in range(len(data_with_cyc_after_temp))[0::10]:
    data_with_cyc_after.append(data_with_cyc_after_temp[i:i+10])

num_batch = len(data_with_cyc_after_output_temp)/batch_size


for k in range(num_batch):
    temp1 = []
    for j in range(10):
        temp = []
        for i in range(batch_size*10)[0::10]:
            temp.append(data_with_cyc_after_output_temp[j])
        temp1.append(temp)
    data_with_cyc_after_output.append(temp1)










# Parameters
learning_rate = 0.0005
# training_iters = 12000
training_iters = 500000
display_step = 100 

# Network Parameters
n_input = 6 # MNIST data input (img shape: 28*28) columns
n_steps = 10 # timesteps
n_hidden = 64 # hidden layer num of features
n_classes = 2 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [n_steps, None, 2])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]),trainable = training)
}
biases = {
    'out': tf.Variable(tf.random_normal([2]),trainable = training)
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    out_rnn = []
    for i in range(n_steps):
        out_rnn.append(tf.matmul(outputs[i],weights['out']) + biases['out'])
    # Linear activation, using rnn inner loop last output
    return out_rnn

def input_data(steps):
    batch_x = data_with_cyc_after[(steps - 1) * batch_size:steps * batch_size]
    batch_y = data_with_cyc_after_output[steps - 1]
    return batch_x, batch_y # batch_x 128*28*28 batch_y 128*10



pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_sum(tf.square(pred - y)) #added by fan
saver = tf.train.Saver()

#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
#correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = input_data(step)
            # Reshape data to get 28 seq of 28 elements
            # batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            # Run optimization op (backprop)
        if training:
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            # saver.save(sess,"./model/model.ckpt")
            if step % display_step == 0:
                # Calculate batch accuracy
                # acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss))
                #print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                #      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                #     "{:.5f}".format(acc))
            step += 1
        else:
            saver.restore(sess, "./model/model.ckpt")
            # saver.restore(sess, "./train/model.ckpt")
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            #saver.save(sess,"./model/model.skpt")
            if step % display_step == 0:
                # Calculate batch accuracy
                # acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss))
                #print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                #      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                #     "{:.5f}".format(acc))
            step += 1
    if training:
        saver.save(sess,"./model/model.ckpt")

    print("Optimization Finished!")

