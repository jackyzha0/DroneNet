import dataset
import numpy as np
db = dataset.dataHandler(train = "data/training", test="data/testing", NUM_CLASSES = 4)
print(db)
print(len(db.train_unused))
img, label = db.minibatch(2)
dataset.dispImage(img[0], boundingBoxes = label[0], drawTime = 5000)
