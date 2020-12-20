import tensorflow as tf
import numpy as np

# 构造数据
x_data = np.random.rand(100)
y_data = 0.1 * x_data + 0.2

# 构造计算图
# 1. 定义模型结构（前向传播）
k = tf.Variable(0.)
b = tf.Variable(0.)
y = k * x_data + b

# 2. 定义loss，优化器，最小化loss
loss = tf.reduce_mean(tf.square(y_data - y))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化参数
init = tf.global_variables_initializer()

# 开始训练
with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print(sess.run([k, b]))
