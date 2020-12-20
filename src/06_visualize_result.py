import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# 伪造样本数据
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) + noise + 0.5

# 输入输出数据
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 构造神经网络
L1 = add_layer(x, 1, 10, tf.nn.relu)
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

    # 将预测结果以红线的方式画在样本点上
    plt.figure()
    plt.scatter(x_data, y_data)
    prediction_value = sess.run(prediction, feed_dict={x:x_data})
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()
