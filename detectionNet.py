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
batchsize = 128
epochs = 1200
learning_rate = 5e-4
momentum = 0.9
#sx_dims = 120x120
sx = 7 #448
sy = 7 #448
B = 2 #num bounding boxes per anchor box
C = 5 #class probabilities, size num_classes
outdims = (sx, sy, (B * 5 + C)) #S x S x (B*5 + C)
#Mult by 5 for output shape
# x, y, w, h, confidence

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

images = tf.placeholder(tf.float32, [None, 448, 448, 3], name="x_inp")
boxes = tf.placeholder(tf.float32, [None, None, (5 + C)], name="y_inp")

net = tf.contrib.layers.conv2d(images, 96, [7, 7], stride=2, scope='conv1')
net = tf.contrib.layers.max_pool2d(net, [3, 3], stride=2, scope='maxpool1')
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
net = tf.contrib.layers.max_pool2d(net, [9, 9], stride=3, scope='maxpool9')
pred = tf.contrib.layers.conv2d(net, 20, [1, 1], stride=1, scope='conv2')
variables_names = [v.name for v in tf.trainable_variables()]
init = tf.global_variables_initializer()

#YOLO loss
# loss = tf.constant(1.23)
# optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=momentum, epsilon=1.0)
# train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())

def miniBatch(dir, ind_arr, size = 128):
    fr_arr = []
    for i in range(size):
        fr_arr.append()

framenum = np.random.randint(0,2400)

dims = (448,448)
name = "VIRAT_S_050203_09_001960_002083"
dir = "data/videos/" + name + ".mp4"
t_dir = "data/annotations/" + name + ".viratdata.objects.txt"
ev = dataset.getEvents(t_dir,scale = dims)
_ev = np.array(dataset.getEventFrame(framenum, ev))
fr = dataset.getFrame(dir,framenum)
crop = dataset.crop(fr,dims)
dataset.dispImage(crop, framenum, boundingBoxes = _ev, drawTime=1000, debug = True)

print('X shape:', crop.shape)
print('Y shape:', _ev.shape)

dbform = event.toFeedFormat(_ev)
print('Y_feed shape:', np.array(dbform).shape)

feed = [[crop],[dbform]]



with tf.Session() as sess:
    sess.run(init)

    values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
        print "Variable: ", k
        #print "Shape: ", v.shape
    print(len(values))

    out = sess.run([pred], feed_dict={images: feed[0], boxes: feed[1]})
    print(np.array(out[0]).shape)
        #_, cost, acc = sess.run([train_op, model_func, acc])
