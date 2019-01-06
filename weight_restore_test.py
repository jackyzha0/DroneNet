import tensorflow as tf
import glob

restore_path = "saved_models/"
WRITE_DEBUG = False
RESTORE_SAVED = False

tf.reset_default_graph()
meta_list = [f for f in glob.glob(restore_path+"*.meta")]
print(meta_list)
meta_name = meta_list[-1]
imported_meta = tf.train.import_meta_graph(meta_name)
with tf.Session() as sess:
    imported_meta.restore(sess, tf.train.latest_checkpoint(restore_path))
