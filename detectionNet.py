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
import stats

from tensorflow.python import pywrap_tensorflow
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import conv2d, avg_pool2d, max_pool2d

@add_arg_scope
def conv2d(inputs, filters, kernel_size, stride=1, idx=None, training = None, alpha=0.1, bn = True, trainable = True):
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
    #with tf.name_scope('conv2d'):
    size = kernel_size[0]
    channels = int(inputs.get_shape()[3])
    initializer = tf.contrib.layers.xavier_initializer_conv2d() #Weight initializer

    #Define weight and bias variables
    #weight = tf.Variable(initializer(shape=[size, size, channels, filters]))
    weight = tf.Variable(tf.truncated_normal([size, size, channels, filters], stddev=0.1), trainable = trainable)
    biases = tf.Variable(tf.constant(0.1, shape=[filters]), trainable = trainable)

    #Input padding
    pad_size = size // 2
    pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
    inputs_pad = tf.pad(inputs, pad_mat)

    #Conv function
    conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID', name=idx+'_conv')
    conv_biased = tf.add(conv, biases, name=idx+'_conv_biased')

    #Leaky ReLU
    conv_biased = tf.maximum((alpha * conv_biased), conv_biased, name=idx+'_leaky_relu')

    #Batch Norm
    if bn:
        conv_biased = tf.layers.batch_normalization(conv_biased, training=training, momentum = 0.9, epsilon=1e-5)
        # conv_biased = tf.cond(training, true_fn=lambda: tf.layers.batch_normalization(conv_biased, training=True),
        #                       false_fn=lambda: tf.layers.batch_normalization(conv_biased, training=False))

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

def fc_layer(inputs, hiddens, flat=False, linear=False, trainable=False, training=None, alpha=0.1, bn = True, name=None):
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
        weight = tf.Variable(tf.zeros([dim, hiddens]), trainable=trainable)#tf.Variable(tf.truncated_normal([dim, hiddens], stddev = 1e-1, mean=0.), trainable=trainable) #Works well empircally
        biases = tf.Variable(tf.constant(0.1, shape=[hiddens]), trainable=trainable)

        #Linear activation
        ip = tf.add(tf.matmul(inputs_processed, weight), biases, name=name)

        #Use Batch norm if bn==true
        if bn:
            ip = tf.layers.batch_normalization(ip, training=training, momentum = 0.9, epsilon=1e-5)

        if linear: #Check if linear
            return ip #Linear
        else:
            return tf.maximum(alpha * ip, ip) #Leaky ReLu

#Network contruction
def ret_net(images, nettype, train = False, sx = 6, sy = 6, B = 3, C = 4):

    #Constant declarations
    dim_mul = sx * sy
    dim_mul_B = dim_mul * B

    if nettype == 0:
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

        net = fc_layer(net, 1024, flat=True, linear=False, trainable=True, training = train)
        #net = tf.layers.dropout(net, rate=0.5, training = train) #Dropout
        net = fc_layer(net, 4096, flat=False, linear=False, trainable=True, training = train)
        net = fc_layer(net, dim_mul_B*(C+5), flat=False, linear=True, trainable=True, training = train)

    elif nettype == 1:
        net = conv2d(images, 16, [3, 3], stride=1, name='conv1', training = train)
        net = tf.contrib.layers.max_pool2d(net, [2, 2], stride=2, padding="SAME", scope='maxpool2')
        print(net.get_shape())
        net = conv2d(net, 32, [3, 3], stride=1, name='conv3', training = train)
        net = tf.contrib.layers.max_pool2d(net, [2, 2], stride=2, padding="SAME", scope='maxpool4')
        print(net.get_shape())
        net = conv2d(net, 64, [3, 3], stride=1, name='conv5', training = train)
        net = tf.contrib.layers.max_pool2d(net, [2, 2], stride=2, padding="SAME", scope='maxpool6')
        print(net.get_shape())
        net = conv2d(net, 128, [3, 3], stride=1, name='conv7', training = train)
        net = tf.contrib.layers.max_pool2d(net, [2, 2], stride=2, padding="SAME", scope='maxpool8')
        print(net.get_shape())
        net = conv2d(net, 256, [3, 3], stride=1, name='conv9', training = train)
        net = tf.contrib.layers.max_pool2d(net, [2, 2], stride=2, padding="SAME", scope='maxpool10')
        print(net.get_shape())
        net = conv2d(net, 512, [3, 3], stride=1, name='conv11', training = train)
        net = tf.contrib.layers.max_pool2d(net, [2, 2], stride=2, padding="SAME", scope='maxpool12')
        print(net.get_shape())
        net = conv2d(net, 1024, [3, 3], stride=1, name='conv13', training = train)
        net = tf.contrib.layers.max_pool2d(net, [2, 2], stride=2, padding="SAME", scope='maxpool14')
        print(net.get_shape())
        net = conv2d(net, 256, [3, 3], stride=1, name='conv15', training = train)
        net = conv2d(net, 512, [3, 3], stride=2, name='conv16', training = train)

        net = fc_layer(net, 1024, flat=True, linear=False, trainable=True, training = train)
        #net = tf.layers.dropout(net, rate=0.5, training = train) #Dropout
        net = fc_layer(net, 4096, flat=False, linear=False, trainable=True, training = train)
        net = fc_layer(net, dim_mul_B*(C+5), flat=False, linear=True, trainable=True, training = train)
    else:
        net = conv2d(images, 64, [7, 7], stride=2, idx='2', training = train, trainable=True)

        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name='maxpool3')

        net = conv2d(net, 192, [3, 3], stride=1, idx='4', training = train, trainable=True)
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name='maxpool5')

        net = conv2d(net, 128, [1, 1], stride=1, idx='6', training = train, trainable=True)
        net = conv2d(net, 256, [3, 3], stride=1, idx='7', training = train, trainable=True)
        net = conv2d(net, 256, [1, 1], stride=1, idx='8', training = train, trainable=True)
        net = conv2d(net, 512, [3, 3], stride=1, idx='9', training = train, trainable=True)
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name='maxpool9')

        net = conv2d(net, 256, [1, 1], stride=1, idx='11', training = train, trainable=True)
        net = conv2d(net, 512, [3, 3], stride=1, idx='12', training = train, trainable=True)
        net = conv2d(net, 256, [1, 1], stride=1, idx='13', training = train, trainable=True)
        net = conv2d(net, 512, [3, 3], stride=1, idx='14', training = train, trainable=True)
        net = conv2d(net, 256, [1, 1], stride=1, idx='15', training = train, trainable=True)
        net = conv2d(net, 512, [3, 3], stride=1, idx='16', training = train, trainable=True)
        net = conv2d(net, 256, [1, 1], stride=1, idx='17', training = train, trainable=True)
        net = conv2d(net, 512, [3, 3], stride=1, idx='18', training = train, trainable=True)
        net = conv2d(net, 512, [1, 1], stride=1, idx='19', training = train, trainable=True)
        net = conv2d(net, 1024, [3, 3], stride=1, idx='20', training = train, trainable=True)
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name='maxpool20')

        net = conv2d(net, 512, [1, 1], stride=1, idx='22', training = train, trainable=True)
        net = conv2d(net, 1024, [3, 3], stride=1, idx='23', training = train, trainable=True)
        net = conv2d(net, 512, [1, 1], stride=1, idx='24', training = train, trainable=True)
        net = conv2d(net, 1024, [3, 3], stride=1, idx='25', training = train, trainable=True)

        net = conv2d(net, 1024, [3, 3], stride=1, idx='26', training = train)
        net = conv2d(net, 1024, [3, 3], stride=2, idx='29', training = train)
        net = conv2d(net, 1024, [3, 3], stride=1, idx='30', training = train)

        net = fc_layer(net, 512, flat=True, linear=False, trainable=True, training = train, name='fc33')
        #net = tf.layers.dropout(net, rate=0.5, training = train) #Dropout
        net = fc_layer(net, 4096, flat=False, linear=False, trainable=True, training = train, name='fc34')
        net = fc_layer(net, dim_mul_B*(C+5), flat=False, linear=True, trainable=True, training = train, name='fc35')
        return net

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

def print_tensors_in_checkpoint_file(file_name, tensor_name, all_tensors):
    varlist=[]
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors:
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in sorted(var_to_shape_map):
        varlist.append(key)
    return varlist
