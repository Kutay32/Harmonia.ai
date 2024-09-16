import cv2
import numpy as np

from data_loader import masks_array, images_array
from keras.preprocessing import image

def resize_images(images, masks, target_size=(128, 128)):
    resized_images = []
    resized_masks = []

    # Loop through each image and its corresponding mask
    for image, mask in zip(images, masks):
        # Resize image to target size using INTER_AREA (good for downscaling)
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        # Resize mask to target size using INTER_NEAREST (keeps label boundaries)
        resized_mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

        # Append resized image and mask to their respective lists
        resized_images.append(resized_image)
        resized_masks.append(resized_mask)

    # Convert the list of resized images and masks into NumPy arrays
    return np.array(resized_images), np.array(resized_masks)


# Set target size to 128x128 and resize images and masks
target_size = (128, 128)
resized_images, resized_masks = resize_images(images_array, masks_array, target_size)