'''
Tensorflow Neural Network for human detection in aerial images
'''
import cv2 as cv
import sugartensor as tf
import dataset
import random
import numpy as np

### PARAMETERS ###
batchsize = 128
epochs = 1200
learning_rate = 5e-4
width = 1080
height = 1920
#sx_dims = 120x120
sx = 16 #1920/120
sy = 9 #1080/120
B = 2 #num bounding boxes per anchor box
C = 5 #class probabilities, size num_classes
outdims = (sx, sy, (B * 5 + C)) #S x S x (B*5 + C)
#Mult by 5 for output shape
# x, y, w, h, confidence

def fire_module(inputs,
                squeeze_depth,
                expand_depth,
                reuse=None,
                scope=None):
    with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse):
        with arg_scope([conv2d, max_pool2d]):
            net = _squeeze(inputs, squeeze_depth)
            net = _expand(net, expand_depth)
        return net


def _squeeze(inputs, num_outputs):
    return conv2d(inputs, num_outputs, [1, 1], stride=1, scope='squeeze')


def _expand(inputs, num_outputs):
    with tf.variable_scope('expand'):
        e1x1 = conv2d(inputs, num_outputs, [1, 1], stride=1, scope='1x1')
        e3x3 = conv2d(inputs, num_outputs, [3, 3], scope='3x3')
    return tf.concat([e1x1, e3x3], 1)

net = conv2d(images, 96, [7, 7], stride=2, scope='conv1')
net = max_pool2d(net, [3, 3], stride=2, scope='maxpool1')
net = fire_module(net, 16, 64, scope='fire2')
net = fire_module(net, 16, 64, scope='fire3')
net = fire_module(net, 32, 128, scope='fire4')
net = max_pool2d(net, [3, 3], stride=2, scope='maxpool4')
net = fire_module(net, 32, 128, scope='fire5')
net = fire_module(net, 48, 192, scope='fire6')
net = fire_module(net, 48, 192, scope='fire7')
net = fire_module(net, 64, 256, scope='fire8')
net = max_pool2d(net, [3, 3], stride=2, scope='maxpool8')
net = fire_module(net, 64, 256, scope='fire9')
net = conv2d(net, outdims, [1, 1], stride=1, scope='conv10')
net = avg_pool2d(net, [13, 13], stride=1, scope='avgpool10')
logits = tf.squeeze(net, [2], name='logits')

def miniBatch(dir, ind_arr, size = 128):
  fr_arr = []
  for i in range(size):
    fr_arr.append()

name = "VIRAT_S_050203_09_001960_002083"
dir = "data/videos/" + name + ".mp4"
t_dir = "data/annotations/" + name + ".viratdata.objects.txt"
ev = np.array([dataset.getEvents(t_dir)])
rnd = random.randint(0,dataset.getRange(dir))
fr = np.array([dataset.getFrame(dir, rnd)])
bw = cv.cvtColor(cv.cvtColor(fr[0], cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2RGB)
dataset.dispImage(bw, rnd, boundingBoxes = ev[0], drawTime=5000, debug = True)

feed = fr,ev[:][:5]
print(ev[:][:5].shape)
print(feed[1][0][:5].shape)

with tf.Session() as session:
  session.run(init)
  _, cost, acc, pred = session.run([train_step, cross_entropy, accuracy, y_conv],feed_dict={x: feed[0], y_: [feed[1][0][:5]], keep_prob: 1.0})
  print(cost, acc, pred)
