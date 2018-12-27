'''
Tensorflow Neural Network for human detection in aerial images
'''
import cv2 as cv
import sugartensor as tf
import dataset
import random
import numpy as np
import event

from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import conv2d, avg_pool2d, max_pool2d

### PARAMETERS ###
batchsize = 32
epochs = 100
learning_rate = 1e-3
momentum = 0.9
#sx_dims = 120x120
sx = 5 #448
sy = 5 #448
B = 3 #num bounding boxes per anchor box
C = 4 #class probabilities, size num_classes
lambda_coord = 5.0
lambda_no_obj = 0.5

@add_arg_scope
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

def fire_module(inputs, squeeze_depth, expand_depth, scope=None, reuse=None):
    with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse):
        with arg_scope([conv2d, max_pool2d]):
            net = _squeeze(inputs, squeeze_depth)
            net = _expand(net, expand_depth)
            return net

def _squeeze(inputs, num_outputs):
    return conv2d(inputs, num_outputs, [1, 1], name='squeeze')


def _expand(inputs, num_outputs):
    e1x1 = conv2d(inputs, num_outputs, [1, 1], name='e1x1')
    e3x3 = conv2d(inputs, num_outputs, [3, 3], name='e3x3')
    return tf.concat([e1x1, e3x3], 3)

images = tf.placeholder(tf.float32, [None, 375, 375, 3], name="im_inp")
#labels = tf.placeholder(tf.float32, [None, sx, sy, B * (5 + C)], name="y_inp")
x = tf.placeholder(tf.float32, [None, sx, sy, B], name="x_inp")
y = tf.placeholder(tf.float32, [None, sx, sy, B], name="y_inp")
w = tf.placeholder(tf.float32, [None, sx, sy, B], name="w_inp")
h = tf.placeholder(tf.float32, [None, sx, sy, B], name="h_inp")
conf = tf.placeholder(tf.float32, [None, sx, sy, B], name="conf_inp")
probs = tf.placeholder(tf.float32, [None, sx, sy, B*C], name="conf_inp")

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
variables_names = [v.name for v in tf.trainable_variables()]

### Definining Cost
# Label Extraction
obj = tf.constant(1., shape=[batchsize, sx, sy, B])
objI = tf.constant(1., shape=[batchsize, sx, sy])
no_obj = tf.constant(0., shape=[batchsize, sx, sy, B])

# Output Extraction
tfBatch = tf.shape(x)[0]
x_, y_, w_, h_, conf_, prob_ = tf.split(net, [B, B, B, B, B, B * C], 3)

subX = tf.subtract(x_, x)
subY = tf.subtract(y_, y)
subW = tf.subtract(tf.sqrt(tf.abs(w_)), tf.sqrt(w))
subH = tf.subtract(tf.sqrt(tf.abs(h_)), tf.sqrt(h))
subC = tf.subtract(conf_, conf)
subP = tf.subtract(prob_, probs)
lossX=tf.multiply(lambda_coord,tf.reduce_sum(tf.multiply(obj,tf.multiply(subX, subX)),axis=[1,2,3]))
lossY=tf.multiply(lambda_coord, tf.reduce_sum(tf.multiply(obj, tf.multiply(subY, subY)),axis=[1,2,3]))
lossW=tf.multiply(lambda_coord, tf.reduce_sum(tf.multiply(obj, tf.multiply(subW, subW)),axis=[1,2,3]))
lossH=tf.multiply(lambda_coord, tf.reduce_sum(tf.multiply(obj, tf.multiply(subH, subH)),axis=[1,2,3]))
lossCObj=tf.reduce_sum(tf.multiply(obj, tf.multiply(subC, subC)),axis=[1,2,3])
lossCNobj=tf.multiply(lambda_no_obj, tf.reduce_sum(tf.multiply(no_obj, tf.multiply(subC, subC)),axis=[1,2,3]))
lossP=tf.reduce_sum(tf.multiply(objI,tf.reduce_sum(tf.multiply(subP, subP), axis=3)) ,axis=[1,2])
loss = tf.add_n((lossX,lossY,lossW,lossH,lossCObj,lossCNobj,lossP))
loss = tf.reduce_mean(loss)

optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=momentum, epsilon=1.0)
train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())

init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()

db = dataset.dataHandler(train = "data/training", test="data/testing", NUM_CLASSES = 4, B = B, sx = 5, sy = 5)

with tf.Session() as sess:
    sess.run(init_g)
    sess.run(init_l)

    while db.batches_elapsed < 100:
        img, label = db.minibatch(batchsize)

        label = np.array(label)
        x_in = label[:,:,:,:B]
        y_in = label[:,:,:,B:2*B]
        w_in = label[:,:,:,2*B:3*B]
        h_in = label[:,:,:,3*B:4*B]
        conf_in = label[:,:,:,4*B:5*B]
        classes_in = label[:,:,:,5*B:(5+C)*B]

        out = sess.run([train_op, lossX, lossY, lossW, lossH, lossCObj, lossCNobj, lossP], feed_dict={images: img, x: x_in, y: y_in, w: w_in, h: h_in, conf: conf_in, probs: classes_in})
        print(out[1:])
