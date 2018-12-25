import dataset
import numpy as np
db = dataset.dataHandler(train = "data/training", test="data/testing", NUM_CLASSES = 4)
print(db)
print(len(db.train_unused))
print(db.minibatch(2))
