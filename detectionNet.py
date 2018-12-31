'''
Tensorflow Neural Network for human detection in aerial images
'''
import cv2 as cv
import sugartensor as tf
import dataset
import random
import numpy as np
from decimal import Decimal
import os
import subprocess

subprocess.call("./tf_board.sh", shell=True)

from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import conv2d, avg_pool2d, max_pool2d

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

config=tf.ConfigProto()

### PARAMETERS ###
batchsize = 1
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
alpha = 0.1

graph = tf.Graph()
with graph.as_default():

    @add_arg_scope
    def conv2d(inputs, filters, kernel_size, stride=1,
               kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
               bias_initializer=tf.zeros_initializer(),
               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002),
               name=None):
        with tf.name_scope('conv2d'):
            return tf.layers.conv2d(inputs, filters, kernel_size, stride,
              kernel_initializer=kernel_initializer,
              bias_initializer=bias_initializer,
              kernel_regularizer=kernel_regularizer,
              activation=tf.nn.relu,
              name=name,
              padding="same")

    def fire_module(inputs, squeeze_depth, expand_depth, scope=None, reuse=None):
        with tf.name_scope('fire'):
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

    def fc_layer(inputs, hiddens, flat=False, linear=False, trainable=False):
        with tf.name_scope('fc'):
            input_shape = inputs.get_shape().as_list()
            if flat:
                dim = input_shape[1] * input_shape[2] * input_shape[3]
                inputs_transposed = tf.transpose(inputs, (0, 3, 1, 2))
                inputs_processed = tf.reshape(inputs_transposed, [-1, dim])
            else:
                dim = input_shape[1]
                inputs_processed = inputs
            weight = tf.Variable(tf.zeros([dim, hiddens]), trainable=trainable)
            biases = tf.Variable(tf.constant(0.1, shape=[hiddens]), trainable=trainable)
            if linear:
                return tf.matmul(inputs_processed, weight) + biases
            else:
                ip = tf.matmul(inputs_processed, weight) + biases
                return tf.maximum(alpha * ip, ip)

    with tf.name_scope('img_in'):
        images = tf.placeholder(tf.float32, [None, 375, 375, 3], name="im_inp")
    with tf.name_scope('label_in'):
        x = tf.placeholder(tf.float32, [None, sx, sy, B], name="x_inp")
        y = tf.placeholder(tf.float32, [None, sx, sy, B], name="y_inp")
        w = tf.placeholder(tf.float32, [None, sx, sy, B], name="w_inp")
        h = tf.placeholder(tf.float32, [None, sx, sy, B], name="h_inp")
        conf = tf.placeholder(tf.float32, [None, sx, sy, B], name="conf_inp")
        probs = tf.placeholder(tf.float32, [None, sx, sy, B*C], name="probs_inp")
        obj = tf.placeholder(tf.float32, shape=[None, sx, sy, B], name="obj")
        objI = tf.placeholder(tf.float32, shape=[None, sx, sy], name="objI")
        no_obj = tf.placeholder(tf.float32, shape=[None, sx, sy, B], name="no_obj")

    dim_mul = sx * sy
    dim_mul_B = dim_mul * B

    with tf.name_scope('net'):
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
        net = tf.contrib.layers.max_pool2d(net, [3, 3], stride=2, scope='maxpool3')
        net = fire_module(net, 64, 256, scope='fire8')
        net = tf.contrib.layers.conv2d(net, dim_mul*B*(C+5), [1, 1], stride=1, scope='conv2')
        net = tf.layers.average_pooling2d(net, [7,7], strides=1)
        net = fc_layer(net, 512, flat=True, linear=False, trainable=True)
        net = fc_layer(net, 4096, flat=False, linear=False, trainable=True)
        net = fc_layer(net, dim_mul_B*(C+5), flat=False, linear=True, trainable=True)

    bn = tf.shape(x)[0]
    net = tf.reshape(net, (bn, sx, sy, B*(C+5)))

    # Output Extraction
    with tf.name_scope('reshape_ops'):
        x_, y_, w_, h_, conf_, prob_ = tf.split(net, [B, B, B, B, B, B * C], axis=3)
        prob__n = tf.reshape(prob_, (bn, sx, sy, B, C))
        probs_n = tf.reshape(probs, (bn, sx, sy, B, C))
        obj_n = tf.expand_dims(obj, axis=-1)
        probs_n = tf.reshape(probs, (bn, sx, sy, B, C))
        obj_n = tf.expand_dims(obj, axis=-1)

    with tf.name_scope('loss_func'):
        delta = tf.constant(1e-8)
        subX = x - x_
        subY = y - y_
        subW = tf.sqrt(w + delta) - tf.sqrt(tf.abs(w_) + delta)
        subH = tf.sqrt(h + delta) - tf.sqrt(tf.abs(h_) + delta)
        subC = conf - conf_
        subP = probs_n - prob__n
        lossX = lambda_coord * tf.reduce_sum(obj * tf.square(subX), axis=[1, 2, 3])
        lossY = lambda_coord * tf.reduce_sum(obj * tf.square(subY), axis=[1, 2, 3])
        lossW = lambda_coord * tf.reduce_sum(obj * tf.square(subW), axis=[1, 2, 3])
        lossH = lambda_coord * tf.reduce_sum(obj * tf.square(subH), axis=[1, 2, 3])
        lossCObj = tf.reduce_sum(obj * tf.square(subC), axis=[1, 2, 3])
        lossCNobj = lambda_no_obj * tf.reduce_sum(no_obj * tf.square(subC), axis=[1, 2, 3])
        lossP = tf.reduce_sum(obj_n * tf.square(subP), axis=[1, 2, 3, 4])
        lossT = tf.add_n((lossX, lossY, lossW, lossH, lossCObj, lossCNobj, lossP))
        loss = tf.reduce_mean(lossT)
        tf.summary.scalar("lossX", tf.reduce_sum(lossX))
        tf.summary.scalar("lossY", tf.reduce_sum(lossY))
        tf.summary.scalar("lossW", tf.reduce_sum(lossW))
        tf.summary.scalar("lossH", tf.reduce_sum(lossH))
        tf.summary.scalar("lossC_obj", tf.reduce_sum(lossCObj))
        tf.summary.scalar("lossC_no_obj", tf.reduce_sum(lossCNobj))
        tf.summary.scalar("lossP", tf.reduce_sum(lossP))
        tf.summary.scalar("loss", tf.reduce_sum(loss))

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon = 1.0)
        #optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,momentum=momentum,centered=True)
        grads = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads)

    merged = tf.summary.merge_all()

#db = dataset.dataHandler(train = "data/training", test="data/testing", NUM_CLASSES = 4, B = B, sx = 5, sy = 5)
db = dataset.dataHandler(train = "data/overfit_test", test="data/testing", NUM_CLASSES = 4, B = B, sx = 5, sy = 5)

def prettyPrint(loss, db):
    lossString = "Loss: %.2e | " % loss
    batches_elapsed = "Batches elapsed: %d | " % db.batches_elapsed
    epochs_elapsed = "Epochs elapsed: %d | " % db.epochs_elapsed
    epoch_progress = "Epoch Progress: %.2f%% " % (100. * (len(db.train_arr)-len(db.train_unused))/len(db.train_arr))
    return lossString + batches_elapsed + epochs_elapsed + epoch_progress

with tf.Session(graph = graph) as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    train_writer = tf.summary.FileWriter('tf_writer/TRAIN', graph=sess.graph)

    while db.batches_elapsed < 1000:
        img, label = db.minibatch(batchsize)

        label = np.array(label)
        x_in = label[:,:,:,:B]
        y_in = label[:,:,:,B:2*B]
        w_in = label[:,:,:,2*B:3*B]
        h_in = label[:,:,:,3*B:4*B]
        conf_in = label[:,:,:,4*B:5*B]
        classes_in = label[:,:,:,5*B:(5+C)*B]
        np.savetxt('debug/classes.txt', np.reshape(classes_in[0], (25, classes_in.shape[3])))
        obj_in = label[:,:,:,9*B:10*B]
        noobj_in = label[:,:,:,10*B:11*B]
        objI_in = label[:,:,:,33]
        if x_in.shape[0] == 1:
            objI_in = [np.squeeze(objI_in)]
        else:
            objI_in = np.squeeze(objI_in)

        out = sess.run([train_op, loss, net, merged],
                       feed_dict={images: img, x: x_in, y: y_in, w: w_in, h: h_in,
                                  conf: conf_in, probs: classes_in,
                                  obj: obj_in, no_obj: noobj_in, objI: objI_in})

        train_writer.add_summary(out[3], db.batches_elapsed)
        train_writer.flush()
        pred_labels = np.array(out)[2][0]
        sk = np.reshape(pred_labels, (sx*sy, 27))
        #np.savetxt('debug/lb%s_out.txt' % db.batches_elapsed, sk)
        print(prettyPrint(out[1], db))
        #print(sk.shape)
        sk2 = label[0]
        #print(sk2.shape)
        sk2_ = np.reshape(sk2, (sx*sy, 34))
        #np.savetxt('debug/lb%s_act.txt' % db.batches_elapsed, sk2_)
        db.dispImage(img[0], boundingBoxes = label[0], preds = np.reshape(pred_labels, (sx, sy, 27)))
