# Import necessary libraries
from numpy.ma.core import shape
from skimage.feature import shape_index
from torch.onnx.operators import shape_as_tensor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import cv2
import torch.optim as optim
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from tqdm import tqdm

TRAIN_DATA_PATH = "data/lgg-mri-segmentation/kaggle_3m"

images = []
masks = []
for filenames in os.walk(TRAIN_DATA_PATH):
    for filename in filenames[2]:
        if 'mask'in filename:
            masks.append(f'{filenames[0]}/{filename}')
            images.append(f'{filenames[0]}/{filename.replace("_mask", "")}')

df = pd.DataFrame({'image': images, 'mask': masks})
df.head()
df.shape  #it should give the result img_size,2
#df.to_csv('TRAIN_DATA_PATH.csv', index=False)
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

# Select the device to train on
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
EPOCHS = 100        # number of epochs
LR = 0.001         # Learning rate
IMG_SIZE = 320     # Size of image
BATCH_SIZE = 32    # Batch size

# Define pretrained encoder model and weights
ENCODER = 'timm-efficientnet-b0'
WEIGHTS = 'imagenet'

df = pd.DataFrame({'image': images, 'mask': masks})
print(df.shape)
df.head()

def plot_images(images, masks, num_images=10):
    # Create a figure with a specified size
    plt.figure(figsize=(15, num_images * 5))  # Width of 15 and height based on number of images

    for i in range(num_images):
        # Plot image
        plt.subplot(num_images, 2, 2 * i + 1)  # Arrange subplots in a grid with 2 columns
        plt.title('Image')  # Title for the image subplot
        plt.imshow(images[i].squeeze(), cmap='gray')  # Display image in grayscale
        plt.axis('off')  # Hide axis for a cleaner look

        # Plot mask
        plt.subplot(num_images, 2, 2 * i + 2)  # Position the mask subplot next to the image
        plt.title('Mask')  # Title for the mask subplot
        plt.imshow(masks[i].squeeze(), cmap='gray')  # Display mask in grayscale
        plt.axis('off')  # Hide axis for a cleaner look

    plt.tight_layout()  # Adjust subplots to fit in the figure area
    plt.show()  # Display the figure


# Plot a specified number of images and masks
plot_images(images_array, masks_array, num_images=10)

# Split data in separate train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=57)

import albumentations as A

# Define the augmentations
def get_train_augs():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),      # Horizontal Flip with 0.5 probability
        A.VerticalFlip(p=0.5)         # Vertical Flip with 0.5 probability
    ], is_check_shapes=False)

def get_val_augs():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE)
    ], is_check_shapes=False)

from torch.utils.data import Dataset

def get_mask_image_paths(directory, extension=".tiff"):
    mask_image_paths = []
    for filename in os.listdir(directory):
        # Check if the filename ends with the specified extension and contains "_mask"
        if filename.endswith(extension) and "_mask" in filename:
            mask_image_paths.append(os.path.join(directory, filename))
    return mask_image_paths

# Create a custom dataset class
class SegmentationDataset(Dataset):
    def __init__(self, df, augs):
        self.mask = masks
        self.image = images
        self.df = df
        self.augs = augs
        print(f"Initialized with {len(self.image)} images and {len(self.mask)} masks.")
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.image[idx]  # Ensure this is a string path
        mask_path = self.mask[idx]
        images = []
        masks = []
        for filenames in os.walk(TRAIN_DATA_PATH):
            for filename in filenames[2]:
                if 'mask'in filename:
                    masks.append(f'{filenames[0]}/{filename}')
                    images.append(f'{filenames[0]}/{filename.replace("_mask", "")}')
        print(f"Loading image from: {images}, mask from: {masks}")

        # Read images and masks
        image = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = np.expand_dims(masks, axis=-1)

        # Apply augmentations
        if self.augs:
            data = self.augs(image=image, mask=mask)
            image = data['image']
            mask = data['mask']
        # print(f"\nShapes of images after augmentation: {image.shape}")
        # print(f"Shapes of masks after augmentation: {mask.shape}")

        # Transpose image dimensions in pytorch format
        # (H,W,C) -> (C,H,W)
        image = np.transpose(image, (2,0,1)).astype(np.float32)
        mask = np.transpose(mask, (2,0,1)).astype(np.float32)

        # Normalize the images and masks
        image = torch.Tensor(image) / 255.0
        mask = torch.round(torch.Tensor(mask) / 255.0)

        return image, mask

# Processed train and validation sets
train_data = SegmentationDataset(train_df, get_train_augs())
val_data = SegmentationDataset(val_df, get_val_augs())

print(f"Size of Trainset : {len(train_data)}")
print(f"Size of Validset : {len(val_data)}")

def processed_image(idx):
    image, mask = train_data[idx]

    plt.subplot(1,2,1)
    plt.imshow(np.transpose(image, (1,2,0)))
    plt.axis('off')
    plt.title("IMAGE");

    plt.subplot(1,2,2)
    plt.imshow(np.transpose(mask, (1,2,0)), cmap='gray')
    plt.axis('off')
    plt.title("GROUND TRUTH");
    plt.show()

print(f"Number of images in train_data: {len(train_data)}")
train_data = SegmentationDataset(train_df, get_train_augs())
print(f"Image paths: {len(train_df)}, Mask paths: {len(val_data)}")

from torch.utils.data import DataLoader

trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

print(f"Total number of batches in Train Loader: {len(trainloader)}")
print(f"Total number of batches in Val Loader: {len(valloader)}")

class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()

        self.model = smp.UnetPlusPlus(
            encoder_name=ENCODER,      # Choose the encoder you're using (like 'resnet34')
            encoder_weights=WEIGHTS,   # Pretrained weights (you can set it to 'None' if not needed)
            in_channels=1,             # Set input channels to 1 for grayscale (MRI)
            classes=1,                 # Binary segmentation
            activation=None            # No activation at the last layer (you'll use BCEWithLogitsLoss)
        )

    def forward(self, images, masks=None):
        logits = self.model(images)

        if masks is not None:
            # Use Dice loss and BCE loss for binary segmentation
            loss1 = DiceLoss(mode='binary')(logits, masks)
            loss2 = nn.BCEWithLogitsLoss()(logits, masks)
            return logits, loss1 + loss2

        return logits

model = SegmentationModel()
model.to(DEVICE);

def eval_model(data_loader, model):
    total_loss = 0.0
    model.eval()

    # Initialize the criterion (loss function)
    criterion = nn.BCEWithLogitsLoss()

    images_list = []
    masks_list = []
    preds_list = []

    with torch.no_grad():
        for images, masks in tqdm(data_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            # Forward pass to get the model's predictions
            outputs = model(images)  # Get the logits from the model
            loss = criterion(outputs, masks)  # Calculate the loss

            # Store images, masks, and predictions for visualization
            images_list.append(images.cpu())  # Store on CPU for plotting
            masks_list.append(masks.cpu())
            preds_list.append(outputs.cpu())

            total_loss += loss.item()

    # Concatenate lists to return a single tensor for images, masks, and predictions
    return total_loss / len(data_loader), torch.cat(images_list), torch.cat(masks_list), torch.cat(preds_list)

def plot_evaluated_results(images, masks, preds, num_samples=5):
    # Randomly select indices for the samples to plot
    indices = torch.randint(0, images.size(0), (num_samples,))

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    for i, idx in enumerate(indices):
        # Plot original image
        axes[i, 0].imshow(images[idx, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Original Image {idx.item()}')
        axes[i, 0].axis('off')

        # Plot true mask
        axes[i, 1].imshow(masks[idx, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'True Mask {idx.item()}')
        axes[i, 1].axis('off')

        # Plot predicted mask
        # Apply a sigmoid function to the outputs and threshold it
        pred_mask = torch.sigmoid(preds[idx, 0]) > 0.5
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title(f'Predicted Mask {idx.item()}')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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

x_train, x_test, y_train, y_test = train_test_split(resized_images, resized_masks, test_size=0.4, random_state=42)  # 60% train, 40% temp
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)  # 15% test, 15% validation

# Convert data to PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
y_train = torch.tensor(y_train, dtype=torch.float32)
x_val = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1)
y_val = torch.tensor(y_val, dtype=torch.float32)

# Create DataLoader for batching
train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Define model, loss function, and optimizer
model = SegmentationModel()
model.to(DEVICE)
criterion = nn.BCEWithLogitsLoss()  # Use binary cross-entropy with logits for binary segmentation
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
early_stopping_patience = 4

best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    # Training phase
    for images, masks in train_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)  # Move data to device (GPU/CPU)

    # Maskeleri yeniden şekillendirin
        masks = masks.unsqueeze(1)

        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(images)  # Forward pass

    # Ensure the output and mask shapes match
        assert outputs.shape == masks.shape, f"Output shape {outputs.shape} must match mask shape {masks.shape}"

        loss = criterion(outputs, masks)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Validation phase
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)  # Move data to device

        # Maskeleri yeniden şekillendirin
            masks = masks.unsqueeze(1)  # Hedef maskeyi [batch_size, 1, height, width] yapın

            outputs = model(images)

        # Ensure the output and mask shapes match
            assert outputs.shape == masks.shape, f"Output shape {outputs.shape} must match mask shape {masks.shape}"

            loss = criterion(outputs, masks)  # Compute loss
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'predicts/best_model.pt')  # Save the best model
        print(f"Best model saved with validation loss: {best_val_loss:.4f}")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= early_stopping_patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
torch.save(model.state_dict(), 'predicts/best_model.pt')

import numpy as np
import matplotlib.pyplot as plt

# Rastgele 10 görüntü seçin
num_samples = 10
random_indices = np.random.choice(len(x_test), num_samples, replace=False)
selected_images = x_test[random_indices]

# Test görüntülerini PyTorch tensörüne dönüştürün
selected_images_tensor = torch.tensor(selected_images, dtype=torch.float32).unsqueeze(1)  # [num_samples, 1, height, width]

# Cihaz (CPU/GPU) ayarları
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)
selected_images_tensor = selected_images_tensor.to(DEVICE)

# Tahmin yapma
with torch.no_grad():  # Gradient hesaplamasını devre dışı bırak
    predicted_masks = model(selected_images_tensor)  # Model ile tahmin yap
    predicted_masks = torch.sigmoid(predicted_masks)  # BCEWithLogitsLoss kullanıyorsanız sigmoid uygula

# Çıktıyı NumPy dizisine çevirin ve uygun boyutlandırmayı yapın
predicted_masks_np = predicted_masks.cpu().numpy()  # CPU'ya geri döndür ve NumPy dizisine çevir
predicted_masks_np = predicted_masks_np.squeeze(1)  # [num_samples, 1, height, width] -> [num_samples, height, width]

# Görüntüleri ve tahmin edilen maskeleri görselleştirme
fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3 * num_samples))

for i in range(num_samples):
    # `selected_images` ve `predicted_masks_np` boyutlarını kontrol edin
    img = selected_images[i].squeeze()  # [1, height, width] -> [height, width]
    mask = predicted_masks_np[i]  # [height, width]

    axes[i, 0].imshow(img, cmap='gray', vmin=0, vmax=1)  # Orijinal görüntüyü göster
    axes[i, 0].set_title(f'Original Image {random_indices[i]}')
    axes[i, 0].axis('off')

    axes[i, 1].imshow(mask, cmap='gray', vmin=0, vmax=1)  # Tahmin edilen maskeyi göster
    axes[i, 1].set_title(f'Predicted Mask {random_indices[i]}')
    axes[i, 1].axis('off')

plt.tight_layout()
plt.show()


"""
# Evaluate the model and get the results
val_loss, images, masks, preds = eval_model(val_loader, model)
print(f'Validation Loss: {val_loss:.4f}')
# Plot the evaluated results
plot_evaluated_results(images, masks, preds, num_samples=8)
"""
