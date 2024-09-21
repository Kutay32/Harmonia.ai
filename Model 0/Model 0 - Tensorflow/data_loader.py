# For data manipulation & loading
import numpy as np
import pandas as pd
import os
import cv2

data = "data/lgg-mri-segmentation/kaggle_3m"

images = []
masks = []
for filenames in os.walk(data):
    for filename in filenames[2]:
        if 'mask'in filename:
            masks.append(f'{filenames[0]}/{filename}')
            images.append(f'{filenames[0]}/{filename.replace("_mask", "")}')

df = pd.DataFrame({'image': images, 'mask': masks})
df.head()
df.shape  #it should give the result img_size,2

def load_and_preprocess(images, masks):
    image_data = []
    mask_data = []

    # Load images and masks, convert them to grayscale, and append to lists
    for image_path, mask_path in zip(images, masks):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is not None and mask is not None:
            image_data.append(image)
            mask_data.append(mask)

    # Convert lists to NumPy arrays and normalize pixel values to [0, 1]
    image_data = np.array(image_data) / 255.0
    mask_data = np.array(mask_data) / 255.0

    # Add an extra channel dimension to handle grayscale images
    image_data = np.expand_dims(image_data, axis=-1)
    mask_data = np.expand_dims(mask_data, axis=-1)

    return image_data, mask_data


# Preprocess and return arrays of images and masks
images_array, masks_array = load_and_preprocess(images, masks)