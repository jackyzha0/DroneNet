'''
Script to pre-cache image and labels into .npy format
'''

import dataset
import numpy as np
import tensorflow as tf

# Set and print dataset in use (note the omission of NP arg)
db = dataset.dataHandler(train="data/training", val="data/testing", NUM_CLASSES=4, B=3, sx=7, sy=7)
#db = dataset.dataHandler(train = "data/overfit_test_large", val = "data/overfit_test_large", NUM_CLASSES = 4, B = 3, sx = 5, sy = 5)
print(db)

VAL_PERCENT = 0.8  # Percent of dataset to use for training
ind_val = (int(len(db.train_arr) * VAL_PERCENT))

# Split training set into val/train
train_arr = db.train_arr[:ind_val]
val_arr = db.train_arr[ind_val:]

serialized_path = 'serialized_data/'
#serialized_path = 'data/serialized_small/'

# Write Training Data
serialized_train = serialized_path + 'TRAIN/'

for i in range(len(train_arr)):
    imgdir = db.train_img_dir + "/" + train_arr[i] + ".png"
    img, refx = db.ret_img(imgdir)

    refdims = {}
    refdims[train_arr[i]] = [refx, refx + db.IMGDIMS[1]]
    fname = db.train_label_dir + "/" + train_arr[i] + ".txt"
    label = db.ret_label(fname, refdims, train_arr[i])

    img = img.flatten()
    label = label.flatten()

    save_loc_im = serialized_train + "image/" + train_arr[i] + ".npy"
    np.save(save_loc_im, img)

    save_loc_lb = serialized_train + "label/" + train_arr[i] + ".npy"
    np.save(save_loc_lb, label)

    print('Train data: {}/{}'.format(i + 1, len(train_arr)))

print('Train data serialized!')

# Write Validation Data
serialized_train = serialized_path + 'VAL/'

for i in range(len(val_arr)):
    imgdir = db.train_img_dir + "/" + val_arr[i] + ".png"
    img, refx = db.ret_img(imgdir)

    refdims = {}
    refdims[val_arr[i]] = [refx, refx + db.IMGDIMS[1]]
    fname = db.train_label_dir + "/" + val_arr[i] + ".txt"
    label = db.ret_label(fname, refdims, val_arr[i])

    img = img.flatten()
    label = label.flatten()

    save_loc_im = serialized_train + "image/" + val_arr[i] + ".npy"
    np.save(save_loc_im, img)

    save_loc_lb = serialized_train + "label/" + val_arr[i] + ".npy"
    np.save(save_loc_lb, label)

    print('Validation data: {}/{}'.format(i + 1, len(val_arr)))

print('Validation data serialized!')
