import tensorflow as tf

def add_layer(inputs, in_size, out_size, activation_function=None):
    """
    增加全连接层，返回一个 batch_size * out_size 的矩阵
    - inputs: 输入矩阵，形状 batch_size * in_size
    - in_size: 输入特征数
    - out_size: 输出特征数
    - activation_function: 激活函数
    """
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if not activation_function:
        output = Wx_plus_b
    else:
        output = activation_function(Wx_plus_b)
    return output


def test_add_layer():
    inputs = tf.constant([[1,2,3],
                         [3,4,5]], tf.float32)
    ret = add_layer(inputs, 3, 2)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(ret))

test_add_layer()
