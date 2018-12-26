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

images = tf.placeholder(tf.float32, [None, 375, 375, 3], name="x_inp")
labels = tf.placeholder(tf.float32, [None, sx, sy, (5 + C)], name="y_inp")

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

### Definining Cost
# Label Extraction
nboxes = tf.tile(boxes, B)
x = nboxes[:, :, :, :, 0]
y = nboxes[:, :, :, :, 1]
w = nboxes[:, :, :, :, 2]
h = nboxes[:, :, :, :, 3]
conf
prob
obj
objI
no_obj

# Output Extraction
x_
y_
w_
h_
conf_
prob_
obj_
objI_
no_obj_


init = tf.global_variables_initializer()

# optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=momentum, epsilon=1.0)
# train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())

framenum = np.random.randint(0,2400)

db = dataset.dataHandler(train = "data/training", test="data/testing", NUM_CLASSES = 4)
img, label = db.minibatch(batchsize)

feed = [img,label]

with tf.Session() as sess:
    sess.run(init)
    print(net.get_shape())

    out = sess.run([net], feed_dict={images: feed[0], labels: feed[1]})
    #print(np.array(out[0]).shape)
        #_, cost, acc = sess.run([train_op, model_func, acc])
