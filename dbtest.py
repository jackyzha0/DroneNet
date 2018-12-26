import dataset
import numpy as np
db = dataset.dataHandler(train = "data/training", test="data/testing", NUM_CLASSES = 4)
print(db)
print(len(db.train_unused))
img, label = db.minibatch(64)

for a,b in zip(img, label):
    dataset.dispImage(a, boundingBoxes = b, drawTime = 10)
