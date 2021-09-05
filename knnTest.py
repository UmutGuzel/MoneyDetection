import pickle

import cv2 as cv
import numpy as np
from skimage import feature

photo_path = './sample_images_folder/10_TL/20210722_110934.jpg'


def find_value_of_money(predictions):
    number_of_values = {'5 TL': 0,
                        '10 TL': 0,
                        '20 TL': 0,
                        'null': 0}
    biggest_prediction = 0
    biggest_key = 'null'

    for prediction in predictions:
        number_of_values[prediction] += 1

    for key in number_of_values.keys():
        if number_of_values[key] > biggest_prediction:
            biggest_key = key
            biggest_prediction = number_of_values[key]

    return biggest_key


model = pickle.load(open("knnmodel.pkl", "rb"))

image = cv.imread(photo_path)

image_height, image_width = image.shape[:2]

predictions = []
new_image_width = 248
new_image_height = 128
range_step = 100
p, r = (8, 2)

for i in range(10, 101, 100):
    for j in range(0, image_width, range_step):
        for k in range(0, image_height, range_step):
            cropped_image = image[k: int(image_height * (i / 100) + k),
                                  j: int(image_width * (i / 100) + j)]
            resized_cropped_image = cv.resize(cropped_image, (new_image_width, new_image_height))
            gray = cv.cvtColor(resized_cropped_image, cv.COLOR_BGR2GRAY)

            lbp = feature.local_binary_pattern(gray, p, r, method="uniform")

            (histogram, bin_edges) = np.histogram(lbp.ravel(),
                                                  bins=np.arange(0, p + 1),
                                                  range=(0, p + 2))
            histogram = histogram.astype("float")
            histogram /= (histogram.sum() + 1e-6)

            predictions.append(model.predict([histogram])[0])

prediction = find_value_of_money(predictions)
image = cv.resize(image, (841, 400))

cv.putText(image, prediction, (20, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
cv.imshow("Image", image)

print(prediction)

while True:
    if ord('q') == cv.waitKey(1):
        break

cv.destroyAllWindows()
