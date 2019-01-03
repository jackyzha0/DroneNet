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
from datetime import datetime

subprocess.call("./tf_board.sh", shell=True)

from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import conv2d, avg_pool2d, max_pool2d

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
config=tf.ConfigProto()#gpu_options=gpu_options)

### PARAMETERS ###
batchsize = 3
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
               kernel_regularizer=None,#tf.contrib.layers.l2_regularizer(scale=0.0002),
               name=None, activation=tf.nn.relu):
        with tf.name_scope('conv2d'):
            return tf.layers.conv2d(inputs, filters, kernel_size, stride,
              kernel_initializer=kernel_initializer,
              bias_initializer=bias_initializer,
              kernel_regularizer=kernel_regularizer,
              activation=activation,
              padding="same")

    def fire_module(inputs, squeeze_depth, expand_depth, scope=None, reuse=None):
        with tf.name_scope('fire'):
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
            weight = tf.Variable(tf.truncated_normal([dim, hiddens], stddev = 1e-2), trainable=trainable)
            biases = tf.Variable(tf.constant(0., shape=[hiddens]), trainable=trainable)
            if linear:
                return tf.matmul(inputs_processed, weight) + biases
            else:
                ip = tf.matmul(inputs_processed, weight) + biases
                return tf.maximum(alpha * ip, ip)

    with tf.name_scope('img_in'):
        images = tf.placeholder(tf.float32, [None, 375, 375, 3], name="im_inp")
    with tf.name_scope('label_in'):
        x = tf.placeholder(tf.float32, shape=[None, sx, sy, B], name="x_inp")
        y = tf.placeholder(tf.float32, shape=[None, sx, sy, B], name="y_inp")
        w = tf.placeholder(tf.float32, shape=[None, sx, sy, B], name="w_inp")
        h = tf.placeholder(tf.float32, shape=[None, sx, sy, B], name="h_inp")
        conf = tf.placeholder(tf.float32, shape=[None, sx, sy, B], name="conf_inp")
        probs = tf.placeholder(tf.float32, shape=[None, sx, sy, B*C], name="probs_inp")
        obj = tf.placeholder(tf.float32, shape=[None, sx, sy, B], name="obj")
        objI = tf.placeholder(tf.float32, shape=[None, sx, sy], name="objI")
        no_obj = tf.placeholder(tf.float32, shape=[None, sx, sy, B], name="no_obj")

        dim_mul = sx * sy
        dim_mul_B = dim_mul * B

    net = conv2d(images, 96, [7, 7], stride=2, name='conv1')
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
    net = conv2d(net, dim_mul*B*(C+5), [1, 1], stride=1, name='conv2')
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

    with tf.name_scope('label_text_tf_board'):
        #tf_label_x = tf.summary.text("label_x", tf.substr(tf.as_string(tf.reshape(x[0], (sx, sy * B))), pos=0, len=3))
        #tf_label_y = tf.summary.text("label_y", tf.substr(tf.as_string(tf.reshape(y[0], (sx, sy * B))), pos=0, len=3))
        #tf_label_w = tf.summary.text("label_w", tf.substr(tf.as_string(tf.reshape(w[0], (sx, sy * B))), pos=0, len=3))
        #tf_label_h = tf.summary.text("label_h", tf.substr(tf.as_string(tf.reshape(h[0], (sx, sy * B))), pos=0, len=3))
        tf_label_conf = tf.summary.text("label_conf", tf.substr(tf.as_string(tf.transpose(tf.reshape(conf, (bn, sx * sy * B)))), pos=0, len=5))
        tf_label_prob = tf.summary.text("label_prob", tf.substr(tf.as_string(tf.transpose(tf.reshape(probs_n, (bn, sx * sy * B * C)))), pos=0, len=5))
        #tf_label_obj = tf.summary.text("label_obj", tf.substr(tf.as_string(tf.reshape(conf, (bn, sx * sy * B)).T), pos=0, len=3))
        #tf_label_no_obj = tf.summary.text("label_no_obj", tf.substr(tf.as_string(tf.reshape(conf, (bn, sx * sy * B)).T), pos=0, len=3))
        #tf_label = tf.summary.merge([tf_label_x, tf_label_y, tf_label_w, tf_label_h, tf_label_conf, tf_label_prob, tf_label_obj, tf_label_no_obj])
        tf_label = tf.summary.merge([tf_label_conf, tf_label_prob])

    with tf.name_scope('pred_text_tf_board'):
        #tf_pred_x = tf.summary.text("pred_x", tf.substr(tf.as_string(tf.reshape(x_[0], (sx, sy * B))), pos=0, len=3))
        #tf_pred_y = tf.summary.text("pred_y", tf.substr(tf.as_string(tf.reshape(y_[0], (sx, sy * B))), pos=0, len=3))
        #tf_pred_w = tf.summary.text("pred_w", tf.substr(tf.as_string(tf.reshape(w_[0], (sx, sy * B))), pos=0, len=3))
        #tf_pred_h = tf.summary.text("pred_h", tf.substr(tf.as_string(tf.reshape(h_[0], (sx, sy * B))), pos=0, len=3))
        tf_pred_conf = tf.summary.text("pred_conf", tf.substr(tf.as_string(tf.transpose(tf.reshape(conf_, (bn, sx * sy * B)))), pos=0, len=5))
        tf_pred_prob = tf.summary.text("pred_prob", tf.substr(tf.as_string(tf.transpose(tf.reshape(prob__n, (bn, sx * sy * B * C)))), pos=0, len=5))
        #tf_pred = tf.summary.merge([tf_pred_x, tf_pred_y, tf_pred_w, tf_pred_h, tf_pred_conf, tf_pred_prob])
        tf_pred = tf.summary.merge([tf_pred_conf, tf_pred_prob])

    with tf.name_scope('loss_func_tf_board'):
        tf_lossX = tf.summary.scalar("lossX", tf.reduce_sum(lossX))
        tf_lossY = tf.summary.scalar("lossY", tf.reduce_sum(lossY))
        tf_lossW = tf.summary.scalar("lossW", tf.reduce_sum(lossW))
        tf_lossH = tf.summary.scalar("lossH", tf.reduce_sum(lossH))
        tf_lossC_obj = tf.summary.scalar("lossC_obj", tf.reduce_sum(lossCObj))
        tf_lossC_no_obj = tf.summary.scalar("lossC_no_obj", tf.reduce_sum(lossCNobj))
        tf_lossP = tf.summary.scalar("lossP", tf.reduce_sum(lossP))
        tf_loss = tf.summary.scalar("loss", tf.reduce_sum(loss))
        merged = tf.summary.merge([tf_lossX, tf_lossY, tf_lossW, tf_lossH, tf_lossC_obj, tf_lossC_no_obj, tf_lossP, tf_loss])

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, epsilon = 1e-0)
        #optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,momentum=momentum,centered=True)
        grads = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads)

    with tf.name_scope('tboard_outimages'):
        tf_out = tf.placeholder(tf.float32, shape=[None, 375, 375, 3])
        tf_im_out = tf.summary.image("tf_im_out", tf_out, max_outputs=batchsize)

#db = dataset.dataHandler(train = "data/training", test="data/testing", NUM_CLASSES = 4, B = B, sx = 5, sy = 5)
db = dataset.dataHandler(train = "data/overfit_test", test="data/testing", NUM_CLASSES = 4, B = B, sx = 5, sy = 5)

def prettyPrint(loss, db):
    lossString = "Loss: %.3f | " % loss
    batches_elapsed = "Batches elapsed: %d | " % db.batches_elapsed
    epochs_elapsed = "Epochs elapsed: %d | " % db.epochs_elapsed
    epoch_progress = "Epoch Progress: %.2f%% " % (100. * (len(db.train_arr)-len(db.train_unused))/len(db.train_arr))
    return lossString + batches_elapsed + epochs_elapsed + epoch_progress

with tf.Session(graph = graph, config = config) as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    print('tf_writer/TRAIN/%s' % str(datetime.now()).split(' ')[1][:8])
    train_writer = tf.summary.FileWriter('tf_writer/TRAIN/%s' % str(datetime.now()).split(' ')[1][:8], graph=sess.graph)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,output_partition_graphs=True)
    saver = tf.train.Saver()

    while db.batches_elapsed < 1000:
        img, label = db.minibatch(batchsize)

        label = np.array(label)
        x_in = label[:,:,:,:B]
        y_in = label[:,:,:,B:2*B]
        w_in = label[:,:,:,2*B:3*B]
        h_in = label[:,:,:,3*B:4*B]
        conf_in = label[:,:,:,4*B:5*B]
        classes_in = label[:,:,:,5*B:(5+C)*B]
        obj_in = label[:,:,:,9*B:10*B]
        noobj_in = label[:,:,:,10*B:11*B]
        objI_in = label[:,:,:,33]
        if x_in.shape[0] == 1:
            objI_in = [np.squeeze(objI_in)]
        else:
            objI_in = np.squeeze(objI_in)

        out = sess.run([train_op, loss, net, merged, tf_label, tf_pred],
                       feed_dict={images: img, x: x_in, y: y_in, w: w_in, h: h_in,
                                  conf: conf_in, probs: classes_in,
                                  obj: obj_in, no_obj: noobj_in, objI: objI_in}, options=run_options)

        pred_labels = np.array(out)[2]
        print(prettyPrint(out[1], db))
        if db.batches_elapsed % 1000 == 0:#batchsize == 0:
            save_path = saver.save(sess, 'saved_models/save%s' % str(db.batches_elapsed))

        im = np.zeros((batchsize, 375, 375, 3))
        # label0 = np.reshape(label[0], (sx*sy, 34))
        # label1 = np.reshape(label[1], (sx*sy, 34))
        # predlabel0 = np.reshape(pred_labels[0], (sx*sy, 27))
        # predlabel1 = np.reshape(pred_labels[1], (sx*sy, 27))
        # np.savetxt('debug/lb0_act.txt', label0)
        # np.savetxt('debug/lb1_act.txt', label1)
        # np.savetxt('debug/lb0_pred.txt', predlabel0)
        # np.savetxt('debug/lb1_pred.txt', predlabel1)
        for i in range(len(img)):
            im[i] = db.dispImage(img[i], boundingBoxes = label[i], preds = np.reshape(pred_labels[i], (sx, sy, 27)))

        im_tf = sess.run(tf_im_out, feed_dict={tf_out: im})
        train_writer.add_summary(out[3], db.batches_elapsed)
        train_writer.flush()
        if db.batches_elapsed % 32 == 0:
            train_writer.add_summary(im_tf, db.batches_elapsed)
            train_writer.flush()
            train_writer.add_summary(out[4], db.batches_elapsed)
            train_writer.flush()
            train_writer.add_summary(out[5], db.batches_elapsed)
            train_writer.flush()
