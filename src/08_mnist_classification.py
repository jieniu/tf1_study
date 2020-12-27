import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 将MNIST数据(http://yann.lecun.com/exdb/mnist/)下载到./MNIST_data文件夹，无需解压
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# 每批次读取多少的样本
batch_size = 100
# 计算一共有多少批次
n_batch = mnist.train.num_examples // batch_size
# placeholder
x = tf.placeholder(tf.float32, [None, 784]) # 特征形状 n*784
y = tf.placeholder(tf.float32, [None, 10]) # 样本形状 n*10

def add_layer(inputs, in_size, out_size, activation_function=None):
    """
    增加全连接层，返回一个 batch_size * out_size 的矩阵
    - inputs: 输入矩阵，形状 batch_size * in_size
    - in_size: 输入特征数
    - out_size: 输出特征数
    - activation_function: 激活函数
    - init_function: 参数的初始化方法
    """
    Weights = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
    biases = tf.Variable(tf.zeros([1, out_size]))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if not activation_function:
        return Wx_plus_b
    else:
        return activation_function(Wx_plus_b)

prediction = add_layer(x, 784, 10, tf.nn.softmax)
# 代价函数
loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y, prediction, 1))
#loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y))
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
#loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))

# 梯度下降
train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#train = tf.train.AdamOptimizer(1e-2).minimize(loss)

# 变量初始化
init = tf.global_variables_initializer()
# 求预测最大的概率的位置和y进行比较，存放在一个布尔型列表中
# tf.argmax将每行最大的索引作为输出，tfargmax([[.1,.2,.3]]) -> [2]
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# 准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(20):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: batch_xs, y: batch_ys})

        # 计算测试集的accuracy
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("epoch " + str(epoch) + ", accuracy " + str(acc))
