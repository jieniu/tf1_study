import tensorflow as tf

# 伪造样本数据，300个样本点，形状为 300*1
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) + noise + 0.5

# 输入输出数据, None表示任意行
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 构造神经网络，一个隐藏层（10个神经元）
L1 = add_layer(x, 1, 10, tf.nn.relu)
# 一个输出层（1个神经元）
prediction = add_layer(L1, 10, 1)
# loss和优化算法
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - prediction),
                                    reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 初始化参数
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        # 如果你采用批量优化算法，就需要使用placeholder输入数据
        sess.run(train, feed_dict={x:x_data, y:y_data})
        if i % 50 == 0:
            print(sess.run(loss, feed_dict={x:x_data, y:y_data}))
