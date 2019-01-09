'''
Tensorflow Neural Network for human and vehicle detection
'''
import sugartensor as tf
import dataset
import random
import glob
import math
import numpy as np
import os
import subprocess
from datetime import datetime
from stats import stats

#Call Tensorboard
subprocess.call("./tf_board.sh", shell=True)

from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import conv2d, avg_pool2d, max_pool2d

#Set debug level to hide warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

#Set Tensorflow run options
run_metadata = tf.RunMetadata()
options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)
config=tf.ConfigProto()

#Directory to save model weights
restore_path = "saved_models/"

WRITE_DEBUG = True #Whether to write img and txt summaries
RESTORE_SAVED = False #Whether to restore saved weights and continue training

### PARAMETERS ###
batchsize = 32
epochs = 100
learning_rate = 1e-3
momentum = 0.9
sx = 5 #Number of horizontal grid cells
sy = 5 #Nubmer of vertical grid cells
B = 3 #num bounding boxes per anchor box
C = 4 #class probabilities, size num_classes
lambda_coord = 5.0 #Loss weighting for cells with objects
lambda_no_obj = 0.5 #Loss weighting for cells without objects
alpha = 0.1 #ReLu coefficient

graph = tf.Graph()
with graph.as_default():

    @add_arg_scope
    def conv2d(inputs, filters, kernel_size, stride=1, name=None, training = True, alpha=0.):
        '''
        Description:
            Creates 2D convolution kernel with batch normalization and leaky ReLu. Dims of next layer can be determined
            by the following formula: N = ((N(n-1) - K) / S) + 1 where N(n-1) are dims of previous layer, K is
            kernel size, and S is stride
        Input:
            inputs: [rank 4 tf tensor] Conv2D input tensor
            filters: [int] Number of conv filters, sets size of 4th dimension
            kernel_size: [length 2 int array] Describes kernel dimensions
            stride: [int] Stride of kernel
            name: [string] Name for tf.graph
            training: [bool] Training flag for disabling batch norm at test time
            alpha: [float] Alpha coefficient, multiplier for Leaky ReLu
        Output:
            conv_biased: [rank 4 tf tensor] Tensor layer output
        '''
        with tf.name_scope('conv2d'):
            size = kernel_size[0]
            channels = int(inputs.get_shape()[3])
            initializer = tf.contrib.layers.xavier_initializer_conv2d() #Weight initializer

            #Define weight and bias variables
            weight = tf.Variable(initializer(shape=[size, size, channels, filters]))
            biases = tf.Variable(tf.constant(0., shape=[filters]))

            #Input padding
            pad_size = size // 2
            pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
            inputs_pad = tf.pad(inputs, pad_mat)

            #Conv function
            conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID')
            conv_biased = tf.add(conv, biases)

            #Leaky ReLU
            conv_biased = tf.maximum((alpha * conv_biased), conv_biased, name='leaky_relu')

            #Batch Norm
            conv_biased = tf.cond(training, true_fn=lambda: tf.layers.batch_normalization(conv_biased, training=True), false_fn=lambda: conv_biased)

            return conv_biased

    def fire_module(inputs, squeeze_depth, expand_depth, scope=None, training = True):
        '''
        Description:
            Creates 'a fire module' (defined in the SqueezeNet paper: arxiv.org/abs/1602.07360)
            Composed of 3 1x1 conv2d layers followed by 4 1x1 conv2d layers concatenated with 4 3x3 conv2d layers
        Input:
            inputs: [rank 4 tf tensor] Conv2D input tensor
            squeeze_depth: [int] Downsample filter number
            expand_depth: [int] Sets output 4th dimension size
            scope: [string] Name for tf.graph
            training: [bool] Training flag for disabling batch norm at test time
        Output:
            net: [rank 4 tf tensor] Tensor layer output
        '''
        with tf.name_scope(scope):
            with arg_scope([conv2d, max_pool2d]):
                #Squeeze Layer
                net = conv2d(inputs, squeeze_depth, [1, 1], name='squeeze', training = training)

                #Expand layer
                e1x1 = conv2d(net, expand_depth, [1, 1], name='e1x1', training = training)
                e3x3 = conv2d(net, expand_depth, [3, 3], name='e3x3', training = training)

                #Concatenate operation
                net = tf.concat([e1x1, e3x3], 3)
                return net

    def fc_layer(inputs, hiddens, flat=False, linear=False, trainable=False, training=True, alpha=0.1, bn = True):
        '''
        Description:
            Creates fully connected layer (FC) with options for ReLu and batch normalizatoin
        Input:
            inputs: [rank 4 tf tensor] Conv2D input tensor
            hidden: [int] Number of output units
            flat: [bool] Whether to flatten to rank one tensor
            linear: [bool] True to use linear, false to use leaky ReLu
            trainable: [bool] Whether layer weights should be updated
            training: [bool] Training flag for disabling batch norm at test time
            alpha: [float] Alpha coefficient, multiplier for Leaky ReLu
            bn: [bool] Whether to use batch normalization
        Output:
            net: [rank 4 tf tensor] Tensor layer output
        '''
        with tf.name_scope('fc'):
            #Fetch input shape
            input_shape = inputs.get_shape().as_list()

            if flat:
                dim = input_shape[1] * input_shape[2] * input_shape[3] #Get flattened dims
                inputs_transposed = tf.transpose(inputs, (0, 3, 1, 2)) #Preserve order
                inputs_processed = tf.reshape(inputs_transposed, [-1, dim]) #Infer dimensions and flatten
            else:
                dim = input_shape[1]
                inputs_processed = inputs

            #Weight and bias declarations
            weight = tf.Variable(tf.truncated_normal([dim, hiddens], stddev = 1e-1, mean=0.), trainable=trainable) #Works well empircally
            biases = tf.Variable(tf.constant(0., shape=[hiddens]), trainable=trainable)

            #Linear activation
            ip = tf.matmul(inputs_processed, weight) + biases

            #Use Batch norm if bn==true
            ip = tf.cond(training, true_fn=lambda: tf.layers.batch_normalization(ip, training=True), false_fn=lambda: ip)

            if linear: #Check if linear
                return ip #Linear
            else:
                return tf.maximum(alpha * ip, ip) #Leaky ReLu

    #Image placeholder for feed_dict
    with tf.name_scope('img_in'):
        images = tf.placeholder(tf.float32, [None, 375, 375, 3], name="im_inp")

    #Label placeholders for feed_dict
    with tf.name_scope('label_in'):
        x = tf.placeholder(tf.float32, shape=[None, sx, sy, B], name="x_inp")
        y = tf.placeholder(tf.float32, shape=[None, sx, sy, B], name="y_inp")
        w = tf.placeholder(tf.float32, shape=[None, sx, sy, B], name="w_inp")
        h = tf.placeholder(tf.float32, shape=[None, sx, sy, B], name="h_inp")
        conf = tf.placeholder(tf.float32, shape=[None, sx, sy, B], name="conf_inp") #Confidence that object exists
        probs = tf.placeholder(tf.float32, shape=[None, sx, sy, B*C], name="probs_inp") #Class probability distribution
        obj = tf.placeholder(tf.float32, shape=[None, sx, sy, B], name="obj") #Object present
        objI = tf.placeholder(tf.float32, shape=[None, sx, sy], name="objI") #Object in grid
        no_obj = tf.placeholder(tf.float32, shape=[None, sx, sy, B], name="no_obj") #No object present
        train = tf.placeholder(tf.bool, name="train_flag") #Training flag placeholder

        #Constant declarations
        dim_mul = sx * sy
        dim_mul_B = dim_mul * B

    #Network contruction, follows SqueezeNet architecture
    net = conv2d(images, 96, [7, 7], stride=2, name='conv1', training = train)
    net = tf.contrib.layers.max_pool2d(net, [3, 3], stride=2, scope='maxpool1')

    net = fire_module(net, 16, 64, scope='fire1', training = train)
    net = fire_module(net, 16, 64, scope='fire2', training = train)
    net = fire_module(net, 32, 128, scope='fire3', training = train)

    net = tf.contrib.layers.max_pool2d(net, [3, 3], stride=2, scope='maxpool2')

    net = fire_module(net, 32, 128, scope='fire4', training = train)
    net = fire_module(net, 48, 192, scope='fire5', training = train)
    net = fire_module(net, 48, 192, scope='fire6', training = train)
    net = fire_module(net, 64, 256, scope='fire7', training = train)

    net = tf.contrib.layers.max_pool2d(net, [3, 3], stride=2, scope='maxpool3')

    net = fire_module(net, 64, 256, scope='fire8', training = train)

    #Average pooling reduces layer connections while maintaining high accuracy
    net = tf.layers.average_pooling2d(net, [7,7], strides=1)

    net = fc_layer(net, 512, flat=True, linear=False, trainable=True, training = train)
    net = fc_layer(net, 4096, flat=False, linear=False, trainable=True, training = train)
    net = fc_layer(net, dim_mul_B*(C+5), flat=False, linear=True, trainable=True, training = train)

    bn = tf.shape(x)[0] #Get batchsize at run-time
    net = tf.reshape(net, (bn, sx, sy, B*(C+5))) #Reshape flattened output

    #Output Extraction
    with tf.name_scope('reshape_ops'):
        x_, y_, w_, h_, conf_, prob_ = tf.split(net, [B, B, B, B, B, B * C], axis=3) #Split output
        prob__n = tf.reshape(prob_, (bn, sx, sy, B, C)) #Reshape to increase dims
        probs_n = tf.reshape(probs, (bn, sx, sy, B, C))
        obj_n = tf.expand_dims(obj, axis=-1) #Ensuring mask has same dimensionality

    #Define cost function
    with tf.name_scope('loss_func'):
        delta = tf.constant(1e-8) #Constant to ensure loss does not become NaN

        #Subtraction operations
        subX = x - x_
        subY = y - y_
        subW = tf.sqrt(w + delta) - tf.sqrt(tf.abs(w_) + delta) #Square root W/H is used to lessen impact of large W/H differences in small boxes
        subH = tf.sqrt(h + delta) - tf.sqrt(tf.abs(h_) + delta) #being overweighted
        subC = conf - conf_ #Confidences
        subP = probs_n - prob__n #Class probs

        #Square and dimension reduction operations
        lossX = lambda_coord * tf.reduce_sum(obj * tf.square(subX), axis=[1, 2, 3]) #Squared difference, reduce_sum across all except batches
        lossY = lambda_coord * tf.reduce_sum(obj * tf.square(subY), axis=[1, 2, 3]) #Multiply by lambda_coord constant to weight boxes with
        lossW = lambda_coord * tf.reduce_sum(obj * tf.square(subW), axis=[1, 2, 3]) #objects more
        lossH = lambda_coord * tf.reduce_sum(obj * tf.square(subH), axis=[1, 2, 3])
        lossCObj = tf.reduce_sum(obj * tf.square(subC), axis=[1, 2, 3])
        lossCNobj = lambda_no_obj * tf.reduce_sum(no_obj * tf.square(subC), axis=[1, 2, 3])
        lossP = tf.reduce_sum(obj_n * tf.square(subP), axis=[1, 2, 3, 4]) #Increased dimension for class probability

        lossT = tf.add_n((lossX, lossY, lossW, lossH, lossCObj, lossCNobj, lossP)) #Sum losses
        loss = tf.reduce_mean(lossT) #Reduce mean across all examples in minibatch

    #Summary ops to log to TensorBoard
    with tf.name_scope('loss_func_tf_board'):
        tf_lossX = tf.summary.scalar("lossX", tf.reduce_mean(lossX))
        tf_lossY = tf.summary.scalar("lossY", tf.reduce_mean(lossY))
        tf_lossW = tf.summary.scalar("lossW", tf.reduce_mean(lossW))
        tf_lossH = tf.summary.scalar("lossH", tf.reduce_mean(lossH))
        tf_lossC_obj = tf.summary.scalar("lossC_obj", tf.reduce_mean(lossCObj))
        tf_lossC_no_obj = tf.summary.scalar("lossC_no_obj", tf.reduce_mean(lossCNobj))
        tf_lossP = tf.summary.scalar("lossP", tf.reduce_mean(lossP))
        tf_loss = tf.summary.scalar("loss", loss)

        #Merge summaries
        merged = tf.summary.merge([tf_lossX, tf_lossY, tf_lossW, tf_lossH, tf_lossC_obj, tf_lossC_no_obj, tf_lossP, tf_loss])

    #Parameter optimizer
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, epsilon = 1e-1) #Used epsilon described in Inception Net paper (0.1)
        grads = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads) #Apply gradients. Call this operation to actually optimize network

    #Placeholders for image visualization in Tensorboard, not used in training
    with tf.name_scope('tboard_outimages'):
        tf_out = tf.placeholder(tf.float32, shape=[None, 375, 375, 3])
        tf_im_out = tf.summary.image("tf_im_out", tf_out, max_outputs=batchsize)

#Creating dataset
db = dataset.dataHandler(train = "serialized_data/TRAIN", val="serialized_data/VAL", NUM_CLASSES = 4, B = B, sx = 5, sy = 5, useNP = True)

def prettyPrint(loss, db, test_eval = False):
    '''
    Description:
        Returns readable string to describe training performance
    Input:
        db: [dataHandler object] Current dataset in use
        test_eval: [bool] Prints validation progress if test_eval
    Output:
        string: [string] Class of object, "unkwn" if no class
    '''
    if test_eval:
        lossString = "Test Loss: %.3f | " % loss
        epoch_progress = "Validation Progress: %.2f%% " % (100. * (len(db.val_arr)-len(db.val_unused))/len(db.val_arr))
        return lossString + epoch_progress
    else:
        lossString = "Loss: %.3f | " % loss
        batches_elapsed = "Batches elapsed: %d | " % db.batches_elapsed
        epochs_elapsed = "Epochs elapsed: %d | " % db.epochs_elapsed
        epoch_progress = "Epoch Progress: %.2f%% " % (100. * (len(db.train_arr)-len(db.train_unused))/len(db.train_arr))
        return lossString + batches_elapsed + epochs_elapsed + epoch_progress

def write4DData(arr, path, ind):
    '''
    Description:
        Writes a 4D np array to disk in a readable format
    Input:
        arr: [4D np array] Array to be written to disk
        path: Path to write array with $ seperator for indexing
        ind: index to write
    Output:
        None
    '''
    n_split = path.split('$') #Split string with seperator
    name = n_split[0] + str(ind) + n_split[1] #Insert index
    for i in range(len(arr)):
        #Write to file with delimiters
        with open(name+'_'+str(i)+'.txt', 'w') as wfile:
            for d_slice in arr[i]:
                wfile.write('------ Dim separator ------\n')
                np.savetxt(wfile, d_slice, fmt='%.2f', delimiter = '|')

#Begin Tensorflow session
with tf.Session(graph = graph, config = config) as sess:
    #Initialize tensorflow vars
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    #Create TF Summary writer for training
    train_writer = tf.summary.FileWriter('tf_writer/TRAIN/%s' % str(datetime.now()).split(' ')[1][:8], graph=sess.graph)

    #Create TF Summary writer for validation
    val_writer = tf.summary.FileWriter('tf_writer/TEST/%s' % str(datetime.now()).split(' ')[1][:8], graph=sess.graph)

    #Session run options
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,output_partition_graphs=True)
    saver = tf.train.Saver()

    if RESTORE_SAVED: #Check to see if program should restore saved weights
        rest_f = open(restore_path + "epoch_marker")
        #Set db information from marker file
        db.batches_elapsed = int(rest_f.readline())
        db.epochs_elapsed = int(rest_f.readline())

        saver.restore(sess, tf.train.latest_checkpoint(restore_path)) #Restore weights

    prev_epoch = db.epochs_elapsed #Set prev_epoch
    while db.epochs_elapsed < epochs:

        img, label = db.minibatch(batchsize) #Fetch minibatch

        #Seperate labels
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

        #Check if batch only has one item and expand dims if needed
        if x_in.shape[0] == 1:
            objI_in = [np.squeeze(objI_in)]
        else:
            objI_in = np.squeeze(objI_in)

        #One step of training
        out = sess.run([train_op, loss, net, merged],
                       feed_dict={images: img, x: x_in, y: y_in, w: w_in, h: h_in,
                                  conf: conf_in, probs: classes_in,
                                  obj: obj_in, no_obj: noobj_in, objI: objI_in, train: True}, options=run_options, run_metadata=run_metadata)

        #Raw network output
        pred_labels = np.array(out)[2]

        #Output loss information to console
        print(prettyPrint(out[1], db))

        #Fetch image
        im = np.zeros((batchsize, 375, 375, 3))
        for i in range(len(img)):
            #Feed to dispImage to interpret output onto image
            im[i] = db.dispImage(img[i], boundingBoxes = label[i], preds = np.reshape(pred_labels[i], (sx, sy, 27)))

        #Run image summary
        im_tf = sess.run(tf_im_out, feed_dict={tf_out: im})

        #Write all text summaries to Tensorboard
        train_writer.add_summary(out[3], db.batches_elapsed)

        #Local text debug and image writing
        if WRITE_DEBUG and (db.batches_elapsed % 16 == 0 or db.batches_elapsed == 1):
            #write4DData(label, 'debug/lb$_act', db.batches_elapsed)
            #write4DData(pred_labels, 'debug/lb$_pred', db.batches_elapsed)

            #Add image summaries to Tensorboard
            train_writer.add_summary(im_tf, db.batches_elapsed)

        #Flush summaries
        train_writer.flush()

        if db.batches_elapsed % 64 == 0 or db.batches_elapsed == 1:
            t_img, t_label = db.minibatch(batchsize, training = False) #Get validation batch

            #Seperate labels
            t_label = np.array(t_label)
            t_x_in = t_label[:,:,:,:B]
            t_y_in = t_label[:,:,:,B:2*B]
            t_w_in = t_label[:,:,:,2*B:3*B]
            t_h_in = t_label[:,:,:,3*B:4*B]
            t_conf_in = t_label[:,:,:,4*B:5*B]
            t_classes_in = t_label[:,:,:,5*B:(5+C)*B]
            t_obj_in = t_label[:,:,:,9*B:10*B]
            t_noobj_in = t_label[:,:,:,10*B:11*B]
            t_objI_in = t_label[:,:,:,33]

            #Expand dims if batchsize = 1
            if t_x_in.shape[0] == 1:
                t_objI_in = [np.squeeze(t_objI_in)]
            else:
                t_objI_in = np.squeeze(t_objI_in)

            #Run validation batch
            t_out = sess.run([loss, net, merged],
                             feed_dict={images: img, x: t_x_in, y: t_y_in, w: t_w_in, h: t_h_in,
                                        conf: t_conf_in, probs: t_classes_in,
                                        obj: t_obj_in, no_obj: t_noobj_in, objI: t_objI_in, train: False},
                                        options=run_options, run_metadata=run_metadata)

            #Get net out
            t_pred_labels = t_out[1]

            #Pretty print validation out
            print(prettyPrint(t_out[0], db, test_eval=True))

            #Get test images
            t_im = np.zeros((batchsize, 375, 375, 3))
            for i in range(len(img)):
                t_im[i] = db.dispImage(t_img[i], boundingBoxes = t_label[i], preds = np.reshape(t_pred_labels[i], (sx, sy, 27)))

            #Run TF Image summary
            t_im_tf = sess.run(tf_im_out, feed_dict={tf_out: t_im})

            #Add text summaries
            val_writer.add_summary(t_out[2], db.batches_elapsed)
            val_writer.add_summary(t_im_tf, db.batches_elapsed)
            val_writer.flush()

        #Check if new epoch
        if not db.epochs_elapsed == prev_epoch: #If so, evaluate on validation set
            #Print epoch
            print("Epoch %s reached! Saving weights..." %str(db.batches_elapsed))

            #Save weights
            save_path = saver.save(sess, restore_path + 'save{0:06d}'.format(db.batches_elapsed))

            #Save train metadata
            with open(restore_path + "epoch_marker", "w") as f:
                f.write(str(db.batches_elapsed)+"\n")
                f.write(str(db.epochs_elapsed))
            print("Weights saved.")

            prev_epoch += 1

            #Get statistics
            stats = stats(np.reshape(t_pred_labels[i], (sx, sy, 27)), t_label[i], db)
            with open(restore_path + "stats.txt", "a") as f2: #Write to disk
                f2.write(str(stats))
