import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def conv2d(x, W):
    """
    - x: tensor shape [batch, in_height, in_weight, in_channels]
    - W: kernel tensor shape [kernel_height, kernel_weight, in_channels, out_channels]
    # stride [1, x_movement, y_movement, 1]
    # trides[0] = strides[3] = 1
    # padding: A `string` from: "SAME", "VALID"
    """
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def max_pool_2x2(x):
    # ksize [1, x, y, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
batch_size = 100
# 计算一共有多少批次
n_batch = mnist.train.num_examples // batch_size
xs = tf.placeholder(tf.float32, [None, 784]) # 28*28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# n*length*width*hight
x_image = tf.reshape(xs, [-1, 28, 28, 1])

# conv1 layer
# patch 5x5, in_height=1, out_height=32
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])  # 每个卷积核一个偏置值
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output shape 28*28*32
h_pool1 = max_pool_2x2(h_conv1) # 14*14*32

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output shape 14*14*64
h_pool2 = max_pool_2x2(h_conv2) # 7*7*64

# fc1 layer，输入神经元 7*7*64，输出神经元1024
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# [n, 7, 7, 64] -> [n, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# fc2 layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=ys, logits=prediction))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

init = tf.global_variables_initializer()
# 求预测最大的概率的位置和y进行比较，存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(ys, 1), tf.argmax(prediction, 1))
# 准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(10):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.7})

        #train_acc = sess.run(accuracy, feed_dict={xs: mnist.train.images, ys: mnist.train.labels, keep_prob: 1.0})
        test_acc = sess.run(accuracy, feed_dict={xs: mnist.test.images, ys: mnist.test.labels, keep_prob: 1.0})

        print("epoch " + str(epoch) + ", test accuracy " + str(test_acc))
