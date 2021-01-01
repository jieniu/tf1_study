## save weights to file
import tensorflow as tf
W = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name="weights")
b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, "mynet/save_net.ckpt")


## restore from file
import tensorflow as tf

# define the same variable
W = tf.Variable(tf.zeros([2, 3]), dtype=tf.float32, name="weights")
b = tf.Variable(tf.zeros([1, 3]), dtype=tf.float32, name="biases")

# not need init
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "mynet/save_net.ckpt")
    print("weights:", sess.run(W))
    print("biases:", sess.run(b))
