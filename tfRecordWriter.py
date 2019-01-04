import dataset
import numpy as np
import tensorflow as tf
db = dataset.dataHandler(train = "data/training", test="data/testing", NUM_CLASSES = 4, B = 3, sx = 5, sy = 5)
print(db)

VAL_PERCENT = 0.8
ind_val = (int(len(db.train_arr)*VAL_PERCENT))

train_arr = db.train_arr[:ind_val]
val_arr = db.train_arr[ind_val:]

serialized_path = 'serialized_data/'

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

#Write Training Data
serialized_train = serialized_path + 'TRAIN'
train_writer = tf.python_io.TFRecordWriter(serialized_train)

for i in range(len(train_arr)):
    imgdir = db.train_img_dir + "/" + train_arr[i] + ".png"
    img, refx = db.ret_img(imgdir)

    refdims = {}
    refdims[train_arr[i]]= [refx, refx+db.IMGDIMS[1]]
    fname = db.train_label_dir + "/" + train_arr[i] + ".txt"
    label = db.ret_label(fname, refdims, train_arr[i])

    img = img.flatten()
    label = label.flatten()

    feat = {'train/label': _float_feature(label),
            'train/image': _float_feature(img)}
    example = tf.train.Example(features=tf.train.Features(feature=feat))

    train_writer.write(example.SerializeToString())

    print('Train data: {}/{}'.format(i+1, len(train_arr)))

print('Train data serialized!')
train_writer.close()

#Write Validation Data
serialized_val = serialized_path + 'VAL'
val_writer = tf.python_io.TFRecordWriter(serialized_val)

for i in range(len(val_arr)):
    imgdir = db.train_img_dir + "/" + val_arr[i] + ".png"
    img, refx = db.ret_img(imgdir)

    refdims = {}
    refdims[val_arr[i]]= [refx, refx+db.IMGDIMS[1]]
    fname = db.train_label_dir + "/" + val_arr[i] + ".txt"
    label = db.ret_label(fname, refdims, val_arr[i])

    img = img.flatten()
    label = label.flatten()

    feat = {'val/label': _float_feature(label),
            'val/image': _float_feature(img)}
    example = tf.train.Example(features=tf.train.Features(feature=feat))

    val_writer.write(example.SerializeToString())

    print('Val data: {}/{}'.format(i+1, len(val_arr)))

print('Validation data serialized!')
val_writer.close()
