import dataset
import numpy as np
db = dataset.dataHandler(train = "data/overfit_test", test="data/testing", NUM_CLASSES = 4)
print(db)
print(len(db.train_unused))
img, label = db.minibatch(1)

for i in range(len(img)):
    db.dispImage(img[i], boundingBoxes = label[i], drawTime = 5000)
