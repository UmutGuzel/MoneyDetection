import os
import pickle

import cv2 as cv
import numpy as np
from skimage import feature
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

train, train_labels, test, test_labels = ([] for i in range(4))
dataPath = 'data'

p, r = (8, 2)

for (root, dirs, files) in os.walk(dataPath):
    label = os.path.basename(root)
    if label is not dataPath:
        for file in files:
            gray = cv.cvtColor(cv.imread(os.path.join(root, file)), cv.COLOR_BGR2GRAY)

            lbp = feature.local_binary_pattern(gray, p, r, method="uniform")

            (histogram, bin_edges) = np.histogram(lbp.ravel(),
                                                  bins=np.arange(0, p + 1),
                                                  range=(0, p + 2))
            histogram = histogram.astype("float")
            histogram /= (histogram.sum() + 1e-6)

            train.append(histogram)
            train_labels.append(label)

train, test, train_labels, test_labels = train_test_split(train, train_labels, test_size=0.25, random_state=42)

model = KNeighborsClassifier()
model.fit(train, train_labels)

print(cross_val_score(KNeighborsClassifier(), train, train_labels))
print(model.score(test, test_labels))

pickle.dump(model, open("knnmodel.pkl", "wb"))
