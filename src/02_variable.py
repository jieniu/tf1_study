import tensorflow as tf

# 定义一个变量，并初始化为0
state = tf.Variable(0, name="counter")
one = tf.constant(1)

# 每次把stat+1通过一个中间变量赋值给stat
new_value = tf.add(state, one)
# 更新变量
update = tf.assign(state, new_value)

# 初始化所有变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
