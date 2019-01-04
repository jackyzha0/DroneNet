import dataset
import numpy as np
db = dataset.dataHandler(train = "data/training", test="data/testing", NUM_CLASSES = 4, B = 3, sx = 5, sy = 5)
print(db)
img, label = db.minibatch(64)

for i in range(len(img)):
    db.dispImage(img[i], boundingBoxes = label[i], drawTime = 500, test=True)
