import numpy as np

from data_augmenter import resized_images, resized_masks
from data_visualiser_final import display_predictions
# Necessary libraries

from unet_model import unet_model_functional
from data_coeff_iou import dice_coefficient, iou, dice_coef_loss

# Ignore the warnings
import warnings
warnings.filterwarnings('ignore')

# DL Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization,GlobalAveragePooling2D,Input, UpSampling2D, concatenate
from tensorflow.keras.utils import image_dataset_from_directory
from keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import regularizers
#PreTrained Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import Xception
#Image Generator DataAugmentation
from keras.preprocessing import image
from sklearn.model_selection import train_test_split

#Early Stopping
from tensorflow.keras.callbacks import EarlyStopping




#Data Split

x_train, x_test, y_train, y_test = train_test_split(resized_images, resized_masks, test_size=0.4, random_state=42) # 60% train, 40% temp
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)  # 15% test, 15% validation

#Model Call

model = unet_model_functional(input_size=(128, 128, 1))

# Reshape the labels (y_train and y_val) to have the shape (batch_size, height, width, 1)
y_train = y_train.reshape((-1, 128, 128, 1))
y_val = y_val.reshape((-1, 128, 128, 1))


model.compile(optimizer='Adam', loss=dice_coef_loss, metrics=["binary_accuracy", iou, dice_coefficient])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, mode='min')

# Engine Start
history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=16,
                    validation_data=(x_val, y_val),callbacks=[early_stopping])
# Select 10 random images from the test set
test_images = 10
indices = np.random.choice(len(x_test), test_images, replace=False)  # Randomly select indices
x_test = x_test[indices]
y_test = y_test[indices]

# Predict masks for the selected images
predicted_masks = model.predict(x_test)

# Display the original images, true masks, and predicted masks
display_predictions(x_test, y_test, predicted_masks, num_images=test_images)
