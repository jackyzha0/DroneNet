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

serialized_train = serialized_path + 'TRAIN'
serialized_val = serialized_path + 'VAL'

#Write Training Data
writer = tf.python_io.TFRecordWriter(serialized_train)
