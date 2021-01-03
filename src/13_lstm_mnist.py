import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
lr = 1e-4
batch_size = 128

n_inputs = 28 # 每次读入一行
n_steps = 28  # 读完一张图片需要读28次
n_hidden_units = 128 # 隐藏层神经元
n_classes = 10
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, n_classes])

# RNN weights
weights = tf.Variable(tf.truncated_normal([n_hidden_units, n_classes], stddev=0.1))
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))

def RNN(X, weights, biases):
    # (128, 28, 28)
    X_in = tf.reshape(X, [-1, n_steps, n_inputs])

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    # outputs记录的是序列中每一个step的结果
    # final_state记录的是序列中最后一次的结果
    # final_state[0]是cell state
    # final_state[1]是hidden_state
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases)

    return results

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    for epoch in range(6):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run([train_op], feed_dict={
                x: batch_xs,
                y: batch_ys
            })
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print("iter " + str(epoch) + ", test accuracy: " + str(acc))
