import tensorflow as tf
import numpy as np

def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        # 将参数的平均值记录到tensorboard的events的tab页中
        tf.summary.scalar("mean", mean) # 平均值
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev", stddev) # 标准差
        tf.summary.scalar("max", tf.reduce_max(var)) # 最大值
        tf.summary.scalar("min", tf.reduce_min(var)) # 最小值
        # 将参数var显示在tensorboard的直方图tab页中
        tf.summary.histogram("histigram", var)


def add_layer(inputs, in_size, out_size, layer_no, activation_function=None):
    """
    增加全连接层，返回一个 batch_size * out_size 的矩阵
    - inputs: 输入矩阵，形状 batch_size * in_size
    - in_size: 输入特征数
    - out_size: 输出特征数
    - activation_function: 激活函数
    """
    with tf.name_scope("layer_{}".format(layer_no)):
        with tf.name_scope("weights"):
            Weights = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1),
                                  name="w")
            variable_summaries(Weights)
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,
                                 name="b")
            variable_summaries(biases)
        with tf.name_scope("wx_plus_b"):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if not activation_function:
            return Wx_plus_b
        else:
            return activation_function(Wx_plus_b)


# 伪造样本数据
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) + noise + 0.5

# 用一个框把inputs给框起来
with tf.name_scope("inputs"):
    # 输入输出数据
    x = tf.placeholder(tf.float32, [None, 1], name="x_input")
    y = tf.placeholder(tf.float32, [None, 1], name="y_input")

# 构造神经网络
L1 = add_layer(x, 1, 10, 1, tf.nn.relu)
prediction = add_layer(L1, 10, 1, 2)
with tf.name_scope("loss"):
    # loss和优化算法
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - prediction),
                                        reduction_indices=[1]))
    tf.summary.scalar("loss", loss)
with tf.name_scope("train"):
    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 初始化参数
init = tf.global_variables_initializer()

# 合并所有的summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    # 将graph信息写到logs/目录下
    writer = tf.summary.FileWriter("logs/", sess.graph)
    for i in range(200):
        sess.run(train, feed_dict={x:x_data, y:y_data})
        if i % 50 == 0:
            # 每50步打包一次统计信息
            result = sess.run(merged, feed_dict={x:x_data, y:y_data})
            writer.add_summary(result, i)
