import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def add_layer(inputs, in_size, out_size, activation_function=None, keep_prob=1.0):
    """
    增加全连接层，返回一个 batch_size * out_size 的矩阵
    - inputs: 输入矩阵，形状 batch_size * in_size
    - in_size: 输入特征数
    - out_size: 输出特征数
    - activation_function: 激活函数
    - keep_prob: dropout 参数，默认为1表示不dropout
    """
    Weights = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
    biases = tf.Variable(tf.zeros([out_size]))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    z = activation_function(Wx_plus_b) if activation_function else Wx_plus_b
    return tf.nn.dropout(z, keep_prob)

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
batch_size = 100
# 计算一共有多少批次
n_batch = mnist.train.num_examples // batch_size
# placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
# 以多大的概率进行drop_out
keep_prob = tf.placeholder(tf.float32)

# 创建一个简单的神经网络
L1 = add_layer(x, 784, 2000, tf.nn.relu, keep_prob)
L2 = add_layer(L1, 2000, 2000, tf.nn.relu, keep_prob)
prediction = add_layer(L2, 2000, 10, tf.nn.softmax)

# 代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
# 梯度下降
train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
# 变量初始化
init = tf.global_variables_initializer()

# 计算准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(10):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})

	# 计算训练集的准确率和测试集的准确率
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        print("epoch " + str(epoch) + ", train accuracy " + str(train_acc) + ", test accuracy " + str(test_acc))
