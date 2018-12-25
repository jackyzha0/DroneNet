import tensorflow as tf

pred_1 = tf.constant(['1b1x', '1b1y', '1b1w', '1b1h', '1b1conf',
                       '1class1bool', '1class2bool', '1class3bool', '1class4bool', '1class5bool'])

pred_2 = tf.constant(['2b1x', '2b1y', '2b1w', '2b1h', '2b1conf',
                      '2class1bool', '2class2bool', '2class3bool', '2class4bool', '2class5bool'])

pred_3 = tf.constant(['3b1x', '3b1y', '3b1w', '3b1h', '3b1conf',
                      '3class1bool', '3class2bool', '3class3bool', '3class4bool', '3class5bool'])

pred_4 = tf.constant(['4b1x', '4b1y', '4b1w', '4b1h', '4b1conf',
                      '4class1bool', '4class2bool', '4class3bool', '4class4bool', '4class5bool'])

cat1 = tf.stack([pred_1,pred_2])
cat2 = tf.stack([pred_3,pred_4])
fincat = tf.stack([cat1,cat2])
fincat = tf.stack([fincat])

trans2 = tf.reshape(fincat, [-1, tf.pow(tf.shape(fincat)[1],2), tf.shape(fincat)[3]])
x = tf.split(trans2, [1, 1, 1, 1, -1], 1)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(trans2))
print(sess.run(tf.shape(trans2)))
print(sess.run(x))
print(sess.run(tf.shape(x)))
