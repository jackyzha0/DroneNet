from detectionNet import *

# Call Tensorboard
subprocess.call("./tf_board.sh", shell=True)

# Set debug level to hide warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Set Tensorflow run options
run_metadata = tf.RunMetadata()
options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)
config = tf.ConfigProto()

# Directory to save model weights
restore_path = "saved_models/"

WRITE_DEBUG = True  # Whether to write img and txt summaries
RESTORE_SAVED = False  # Whether to restore saved weights and continue training
USE_WARM = True
USE_SQUEEZENET = False
USE_TINY = False
USE_YOLO = True

### PARAMETERS ###
batchsize = 4
iters = 24000
learning_rate = 1e-3
momentum = 0.9
sx = 7  # Number of horizontal grid cells
sy = 7  # Nubmer of vertical grid cells
B = 3  # num bounding boxes per anchor box
C = 4  # class probabilities, size num_classes
lambda_coord = 5.0  # Loss weighting for cells with objects
lambda_no_obj = 0.5  # Loss weighting for cells without objects
alpha = 0.1  # ReLu coefficient

graph = tf.Graph()
with graph.as_default():
    # Image placeholder for feed_dict
    with tf.name_scope('img_in'):
        images = tf.placeholder(tf.float32, [None, 448, 448, 3], name="im_inp")

    # Label placeholders for feed_dict
    with tf.name_scope('label_in'):
        x = tf.placeholder(tf.float32, shape=[None, sx, sy, B], name="x_inp")
        y = tf.placeholder(tf.float32, shape=[None, sx, sy, B], name="y_inp")
        w = tf.placeholder(tf.float32, shape=[None, sx, sy, B], name="w_inp")
        h = tf.placeholder(tf.float32, shape=[None, sx, sy, B], name="h_inp")
        conf = tf.placeholder(tf.float32, shape=[None, sx, sy, B], name="conf_inp")  # Confidence that object exists
        probs = tf.placeholder(tf.float32, shape=[None, sx, sy, B * C], name="probs_inp")  # Class probability distribution
        obj = tf.placeholder(tf.float32, shape=[None, sx, sy, B], name="obj")  # Object present
        objI = tf.placeholder(tf.float32, shape=[None, sx, sy], name="objI")  # Object in grid
        no_obj = tf.placeholder(tf.float32, shape=[None, sx, sy, B], name="no_obj")  # No object present
        train = tf.placeholder(tf.bool, name="train_flag")  # Training flag placeholder

    if USE_SQUEEZENET:
        net = ret_net(images, nettype=0, train=train, sx=sx, sy=sy, B=B, C=C)
    elif USE_TINY:
        net = ret_net(images, nettype=1, train=train, sx=sx, sy=sy, B=B, C=C)
    else:
        net = ret_net(images, nettype=2, train=train, sx=sx, sy=sy, B=B, C=C)

    print("Total trainable parameters:")
    print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

    bn = tf.shape(x)[0]  # Get batchsize at run-time
    net = tf.reshape(net, (bn, sx, sy, B * (C + 5)))  # Reshape flattened output

    # Output Extraction
    with tf.name_scope('reshape_ops'):
        x_, y_, w_, h_, conf_, prob_ = tf.split(net, [B, B, B, B, B, B * C], axis=3)  # Split output
        # prob__n = tf.reshape(prob_, (bn, sx, sy, B, C)) #Reshape to increase dims
        #probs_n = tf.reshape(probs, (bn, sx, sy, B, C))
        # obj_n = tf.expand_dims(obj, axis=-1) #Ensuring mask has same dimensionality

    # Define cost function
    with tf.name_scope('loss_func'):
        delta = tf.constant(1e-8)  # Constant to ensure loss does not become NaN

        # Subtraction operations
        subX = x - x_
        subY = y - y_
        # Square root W/H is used to lessen impact of large W/H differences in small boxes
        subW = tf.sqrt(w + delta) - tf.sqrt(tf.abs(w_) + delta)
        subH = tf.sqrt(h + delta) - tf.sqrt(tf.abs(h_) + delta)  # being overweighted
        subC = conf - conf_  # Confidences
        subP = probs - prob_  # Class probs

        # Square and dimension reduction operations
        # Squared difference, reduce_sum across all except batches
        lossX = lambda_coord * tf.reduce_sum(obj * tf.square(subX), axis=[1, 2, 3])
        # Multiply by lambda_coord constant to weight boxes with
        lossY = lambda_coord * tf.reduce_sum(obj * tf.square(subY), axis=[1, 2, 3])
        lossW = lambda_coord * tf.reduce_sum(obj * tf.square(subW), axis=[1, 2, 3])  # objects more
        lossH = lambda_coord * tf.reduce_sum(obj * tf.square(subH), axis=[1, 2, 3])
        lossCObj = tf.reduce_sum(obj * tf.square(subC), axis=[1, 2, 3])
        lossCNobj = lambda_no_obj * tf.reduce_sum(no_obj * tf.square(subC), axis=[1, 2, 3])
        lossP = tf.reduce_sum(objI * tf.reduce_sum(tf.square(subP), axis=3), axis=[1, 2])
        lossT = tf.add_n((lossX, lossY, lossW, lossH, lossCObj, lossCNobj, lossP))  # Sum losses
        loss = tf.reduce_mean(lossT)  # Reduce mean across all examples in minibatch

    # Summary ops to log to TensorBoard
    with tf.name_scope('loss_func_tf_board'):
        tf_lossX = tf.summary.scalar("lossX", tf.reduce_mean(lossX))
        tf_lossY = tf.summary.scalar("lossY", tf.reduce_mean(lossY))
        tf_lossW = tf.summary.scalar("lossW", tf.reduce_mean(lossW))
        tf_lossH = tf.summary.scalar("lossH", tf.reduce_mean(lossH))
        tf_lossC_obj = tf.summary.scalar("lossC_obj", tf.reduce_mean(lossCObj))
        tf_lossC_no_obj = tf.summary.scalar("lossC_no_obj", tf.reduce_mean(lossCNobj))
        tf_lossP = tf.summary.scalar("lossP", tf.reduce_mean(lossP))
        tf_loss = tf.summary.scalar("loss", loss)

        # Merge summaries
        merged = tf.summary.merge([tf_lossX, tf_lossY, tf_lossW, tf_lossH, tf_lossC_obj, tf_lossC_no_obj, tf_lossP, tf_loss])

    # Parameter optimizer
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        with tf.name_scope('optimizer'):
            #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
            # Used epsilon described in Inception Net paper (0.1)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-0)
            #optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate, momentum = momentum)
            grads = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(grads)  # Apply gradients. Call this operation to actually optimize network

    # Placeholders for image visualization in Tensorboard, not used in training
    with tf.name_scope('tboard_outimages'):
        tf_out = tf.placeholder(tf.float32, shape=[None, 448, 448, 3])
        tf_im_out = tf.summary.image("tf_im_out", tf_out, max_outputs=batchsize)

# Creating dataset
db = dataset.dataHandler(train="serialized_data/TRAIN", val="serialized_data/VAL", NUM_CLASSES=C, B=B, sx=sx, sy=sy, useNP=True)
#db = dataset.dataHandler(train = "data/serialized_small/TRAIN", val="data/serialized_small/TRAIN", NUM_CLASSES = 4, B = B, sx = 5, sy = 5, useNP = True)

# Begin Tensorflow session
with tf.Session(graph=graph, config=config) as sess:
    # Initialize tensorflow vars
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    # Create TF Summary writer for training
    train_writer = tf.summary.FileWriter('tf_writer/TRAIN/%s' % str(datetime.now()).split(' ')[1][:8], graph=sess.graph)

    # Create TF Summary writer for validation
    val_writer = tf.summary.FileWriter('tf_writer/TEST/%s' % str(datetime.now()).split(' ')[1][:8], graph=sess.graph)

    # Session run options
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)

    if RESTORE_SAVED:  # Check to see if program should restore saved weights
        rest_f = open(restore_path + "epoch_marker")
        # Set db information from marker file
        db.batches_elapsed = int(rest_f.readline())
        db.epochs_elapsed = int(rest_f.readline())
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(restore_path))  # Restore weights
    elif USE_WARM:
        CHECKPOINT_NAME = restore_path + 'yolo.ckpt'
        restored_vars = get_tensors_in_checkpoint_file(file_name=CHECKPOINT_NAME)
        tensors_to_load = build_tensors_in_checkpoint_file(restored_vars)
        saver = tf.train.Saver(tensors_to_load)
        saver.restore(sess, CHECKPOINT_NAME)

    prev_epoch = db.epochs_elapsed  # Set prev_epoch
    while db.batches_elapsed < iters:

        img, label = db.minibatch(batchsize)  # Fetch minibatch

        # Seperate labels
        label = np.array(label)
        x_in = label[:, :, :, :B]
        y_in = label[:, :, :, B:2 * B]
        w_in = label[:, :, :, 2 * B:3 * B]
        h_in = label[:, :, :, 3 * B:4 * B]
        conf_in = label[:, :, :, 4 * B:5 * B]
        classes_in = label[:, :, :, 5 * B:(5 + C) * B]
        obj_in = label[:, :, :, 9 * B:10 * B]
        noobj_in = label[:, :, :, 10 * B:11 * B]
        objI_in = label[:, :, :, 33]

        # Check if batch only has one item and expand dims if needed
        if x_in.shape[0] == 1:
            objI_in = [np.squeeze(objI_in)]
        else:
            objI_in = np.squeeze(objI_in)

        # One step of training
        # conv28_w = sess.run(tf.get_default_graph().get_tensor_by_name('yolo/conv_28/weights:0'))
        # write4DData(conv28_w, 'kernel_debug/conv28_w_$', db.batches_elapsed)
        # conv23_w = sess.run(tf.get_default_graph().get_tensor_by_name('yolo/conv_23/weights:0'))
        # write4DData(conv23_w, 'kernel_debug/conv23_w_$', db.batches_elapsed)
        #
        # fc33_w = sess.run(tf.get_default_graph().get_tensor_by_name('yolo/fc_33/weights:0'))
        # with open('kernel_debug/fc33_w' + str(db.batches_elapsed) + '.txt', 'w') as wfile:
        #     np.savetxt(wfile, fc33_w, fmt='%.2f', delimiter = '|')

        out = sess.run([train_op, loss, net, merged],
                       feed_dict={images: img, x: x_in, y: y_in, w: w_in, h: h_in,
                                  conf: conf_in, probs: classes_in,
                                  obj: obj_in, no_obj: noobj_in, objI: objI_in, train: True}, options=run_options, run_metadata=run_metadata)

        # Raw network output
        pred_labels = np.array(out)[2]

        # Output loss information to console
        print(prettyPrint(out[1], db))

        # Fetch image
        im = np.zeros((batchsize, 448, 448, 3))
        for i in range(len(img)):
            # Feed to dispImage to interpret output onto image
            im[i] = db.dispImage(img[i], boundingBoxes=label[i], preds=np.reshape(pred_labels[i], (sx, sy, 27)))

        # Run image summary
        im_tf = sess.run(tf_im_out, feed_dict={tf_out: im})

        # Write all text summaries to Tensorboard
        train_writer.add_summary(out[3], db.batches_elapsed)

        # Local text debug and image writing
        if WRITE_DEBUG and (db.batches_elapsed % 16 == 0 or db.batches_elapsed == 1):
            write4DData(label, 'debug/lb$_act', db.batches_elapsed)
            write4DData(pred_labels, 'debug/lb$_pred', db.batches_elapsed)

            # Add image summaries to Tensorboard
            train_writer.add_summary(im_tf, db.batches_elapsed)

        # Flush summaries
        train_writer.flush()

        if db.batches_elapsed % 32 == 0 or db.batches_elapsed == 1:
            t_img, t_label = db.minibatch(batchsize, training=False)  # Get validation batch

            # Seperate labels
            t_label = np.array(t_label)
            t_x_in = t_label[:, :, :, :B]
            t_y_in = t_label[:, :, :, B:2 * B]
            t_w_in = t_label[:, :, :, 2 * B:3 * B]
            t_h_in = t_label[:, :, :, 3 * B:4 * B]
            t_conf_in = t_label[:, :, :, 4 * B:5 * B]
            t_classes_in = t_label[:, :, :, 5 * B:(5 + C) * B]
            t_obj_in = t_label[:, :, :, 9 * B:10 * B]
            t_noobj_in = t_label[:, :, :, 10 * B:11 * B]
            t_objI_in = t_label[:, :, :, 33]

            # Expand dims if batchsize = 1
            if t_x_in.shape[0] == 1:
                t_objI_in = [np.squeeze(t_objI_in)]
            else:
                t_objI_in = np.squeeze(t_objI_in)

            # Run validation batch
            t_out = sess.run([loss, net, merged],
                             feed_dict={images: img, x: t_x_in, y: t_y_in, w: t_w_in, h: t_h_in,
                                        conf: t_conf_in, probs: t_classes_in,
                                        obj: t_obj_in, no_obj: t_noobj_in, objI: t_objI_in, train: False},
                             options=run_options, run_metadata=run_metadata)

            # Get net out
            t_pred_labels = t_out[1]

            # Pretty print validation out
            print(prettyPrint(t_out[0], db, test_eval=True))

            # Get test images
            t_im = np.zeros((batchsize, 448, 448, 3))
            for i in range(len(img)):
                t_im[i] = db.dispImage(t_img[i], boundingBoxes=t_label[i], preds=np.reshape(t_pred_labels[i], (sx, sy, 27)))

            # Run TF Image summary
            t_im_tf = sess.run(tf_im_out, feed_dict={tf_out: t_im})

            # Add text summaries
            val_writer.add_summary(t_out[2], db.batches_elapsed)
            val_writer.add_summary(t_im_tf, db.batches_elapsed)
            val_writer.flush()

        # Check if new epoch
        if not db.epochs_elapsed == prev_epoch:  # If so, evaluate on validation set
            # Print epoch
            print("Iteration %s reached! Saving weights..." % str(db.batches_elapsed))

            # Save weights
            save_path = saver.save(sess, restore_path + 'save{0:06d}'.format(db.batches_elapsed))

            # Save train metadata
            with open(restore_path + "epoch_marker", "w") as f:
                f.write(str(db.batches_elapsed) + "\n")
                f.write(str(db.epochs_elapsed))
            print("Weights saved.")

            prev_epoch += 1

            # Get statistics
            out_stats = stats.stats(np.reshape(t_pred_labels[i], (sx, sy, 27)), t_label[i], db)
            with open(restore_path + "stats.txt", "a") as f2:  # Write to disk
                f2.write(str(out_stats))
