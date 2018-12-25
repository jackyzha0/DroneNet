import dataset
import numpy as np
db = dataset.dataHandler(train = "data/training", test="data/testing")
print(db)
print(len(db.train_unused))
print(len(db.get_indices(4000)))
