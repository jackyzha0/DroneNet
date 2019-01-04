import tensorflow as tf
sess = tf.Session()

num = tf.range(9)
num = tf.reshape(num, [3,3])

print(sess.run(num))
