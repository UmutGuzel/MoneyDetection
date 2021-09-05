import os

import cv2 as cv

new_image_width = 248
new_image_height = 128
image_folder = "sample_images_folder"
new_image_folder_name = "splitted_sample_images_folder"


def scan(path_of_image, file_name):
    image = cv.imread(path_of_image)
    image_height, image_width = image.shape[:2]

    range_step = 100

    # Create new folder
    new_folder = os.path.join("./", new_image_folder_name)
    if not os.path.isdir(new_folder):
        os.mkdir(new_folder)

    # Create sub folder in the folder
    new_sub_folder = os.path.join(new_folder, file_name)
    if not os.path.isdir(new_sub_folder):
        os.mkdir(new_sub_folder)

    # Split the image different scales and parts. Resize splitted image. Save splitted image to the folder.
    for i in range(10, 101, 10):
        for j in range(0, image_width, range_step):
            for k in range(0, image_height, range_step):
                cropped_image = image[k: int(image_height * (i / 100) + k),
                                      j: int(image_width * (i / 100) + j)]
                resized_cropped_image = cv.resize(cropped_image, (new_image_width, new_image_height))

                new_image_name = os.path.join(new_sub_folder,
                                              file_name +
                                              str(int(j / range_step)) +
                                              str(int(k / range_step) + 1) +
                                              str(int(i / 10)) +
                                              ".jpg")

                cv.imwrite(new_image_name, resized_cropped_image)


for (root, _, files) in os.walk(image_folder):
    if os.path.basename(root) is not image_folder:
        print(files)
        for file in files:
            path = os.path.join(root, file)
            scan(path, root.split("\\")[-1])
