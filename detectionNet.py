'''
Tensorflow Neural Network for human detection in aerial images
'''
import cv2 as cv
import sugartensor as tf
import dataset
import random
import numpy as np
import event

### PARAMETERS ###
batchsize = 32
epochs = 100
learning_rate = 1e-4
momentum = 0.9
#sx_dims = 120x120
sx = 5 #448
sy = 5 #448
B = 3 #num bounding boxes per anchor box
C = 4 #class probabilities, size num_classes
lambda_coord = 5.0
lambda_no_obj = 0.5

def conv2d(inputs, filters, kernel_size, stride=1,
           kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
           bias_initializer=tf.zeros_initializer(),
           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002),
           name=None):
    return tf.layers.conv2d(inputs, filters, kernel_size, stride,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      activation=tf.nn.relu,
      name=name,
      padding="same")

def fire_module(inputs, squeeze_depth, expand_depth, scope):
    with tf.variable_scope(scope, 'fire', [inputs]):
        squeezed = _squeeze(inputs, squeeze_depth)
        net = _expand(squeezed, expand_depth)
        return net

def _squeeze(inputs, num_outputs):
    return conv2d(inputs, num_outputs, [1, 1], name='squeeze')


def _expand(inputs, num_outputs):
    e1x1 = conv2d(inputs, num_outputs, [1, 1], name='e1x1')
    e3x3 = conv2d(inputs, num_outputs, [3, 3], name='e3x3')
    return tf.concat([e1x1, e3x3], axis=3)

def slice_tensor(x, start, end=None):

	if end < 0:
		y = x[...,start:]

	else:
		if end is None:
			end = start
		y = x[...,start:end + 1]

	return y

def yolo_loss(pred, label, lambda_coord, lambda_no_obj):
    #Function written by WojciechMormul
	mask = slice_tensor(label, 5)
	label = slice_tensor(label, sx, sy, 4)

	mask = tf.cast(tf.reshape(mask, shape=(-1, GRID_H, GRID_W, N_ANCHORS)),tf.bool)

	with tf.name_scope('mask'):
		masked_label = tf.boolean_mask(label, mask)
		masked_pred = tf.boolean_mask(pred, mask)
		neg_masked_pred = tf.boolean_mask(pred, tf.logical_not(mask))

	with tf.name_scope('pred'):
		masked_pred_xy = tf.sigmoid(slice_tensor(masked_pred, 0, 1))
		masked_pred_wh = tf.exp(slice_tensor(masked_pred, 2, 3))
		masked_pred_o = tf.sigmoid(slice_tensor(masked_pred, 4))
		masked_pred_no_o = tf.sigmoid(slice_tensor(neg_masked_pred, 4))
		masked_pred_c = tf.nn.softmax(slice_tensor(masked_pred, 5, -1))

	with tf.name_scope('lab'):
		masked_label_xy = slice_tensor(masked_label, 0, 1)
		masked_label_wh = slice_tensor(masked_label, 2, 3)
		masked_label_c = slice_tensor(masked_label, 4)
		masked_label_c_vec = tf.reshape(tf.one_hot(tf.cast(masked_label_c, tf.int32), depth=N_CLASSES), shape=(-1, N_CLASSES))

	with tf.name_scope('merge'):
		with tf.name_scope('loss_xy'):
			loss_xy = tf.reduce_sum(tf.square(masked_pred_xy-masked_label_xy))
		with tf.name_scope('loss_wh'):
			loss_wh = tf.reduce_sum(tf.square(masked_pred_wh-masked_label_wh))
		with tf.name_scope('loss_obj'):
			loss_obj = tf.reduce_sum(tf.square(masked_pred_o - 1))
		with tf.name_scope('loss_no_obj'):
			loss_no_obj = tf.reduce_sum(tf.square(masked_pred_no_o))
		with tf.name_scope('loss_class'):
			loss_c = tf.reduce_sum(tf.square(masked_pred_c - masked_label_c_vec))

		loss = lambda_coord*(loss_xy + loss_wh) + loss_obj + lambda_no_obj*loss_no_obj + loss_c

	return loss

images = tf.placeholder(tf.float32, [None, 375, 375, 3], name="x_inp")
boxes = tf.placeholder(tf.float32, [None, sx, sy, (4 + C)], name="y_inp")

net = tf.contrib.layers.conv2d(images, 96, [7, 7], stride=2, scope='conv1')
net = tf.contrib.layers.max_pool2d(net, [4, 4], stride=2, scope='maxpool1')
net = fire_module(net, 16, 64, scope='fire1')
net = fire_module(net, 16, 64, scope='fire2')
net = fire_module(net, 32, 128, scope='fire3')
net = tf.contrib.layers.max_pool2d(net, [3, 3], stride=2, scope='maxpool2')
net = fire_module(net, 32, 128, scope='fire4')
net = fire_module(net, 48, 192, scope='fire5')
net = fire_module(net, 48, 192, scope='fire6')
net = fire_module(net, 64, 256, scope='fire7')
net = tf.contrib.layers.max_pool2d(net, [3, 3], stride=2, scope='maxpool8')
net = fire_module(net, 64, 256, scope='fire8')
net = tf.contrib.layers.max_pool2d(net, [6, 6], stride=4, scope='maxpool9')
net = tf.contrib.layers.conv2d(net, B*(C+5), [1, 1], stride=1, scope='conv2')
y = tf.reshape(net, shape=(-1, sx, sy, B, C + 5), name='y')

#cost = tf.add([])

init = tf.global_variables_initializer()

#YOLO loss
# optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=momentum, epsilon=1.0)
# train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())

framenum = np.random.randint(0,2400)

db = dataset.dataHandler(train = "data/training", test="data/testing", NUM_CLASSES = 4)
img, label = db.minibatch(batchsize)

feed = [img,label]

with tf.Session() as sess:
    sess.run(init)
    print(net.get_shape())

    out = sess.run([y], feed_dict={images: feed[0], boxes: feed[1]})
    #print(np.array(out[0]).shape)
        #_, cost, acc = sess.run([train_op, model_func, acc])
