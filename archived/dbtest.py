import dataset
import numpy as np
db = dataset.dataHandler(train = "serialized_data/TRAIN", val="serialized_data/VAL", NUM_CLASSES = 4, B = 3, sx = 5, sy = 5, useNP = True)

img, label = db.minibatch(64, training= True)

print(np.array(img).shape, np.array(label).shape)

for i in range(len(img)):
    db.dispImage(img[i], boundingBoxes = label[i], drawTime = 500, test=True)
