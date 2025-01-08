import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import pydicom
import nibabel as nib
import numpy as np
from models.unetpp import UNetPlusPlus  # Import the U-Net++ model

# Custom dataset class for grayscale images
class MedicalDataset(Dataset):
    def __init__(self, image_folder, mask_folder):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(".dcm")]
        self.mask_files = [f for f in os.listdir(mask_folder) if f.endswith(".nii.gz")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        mask_path = os.path.join(self.mask_folder, self.mask_files[idx])

        # Load DICOM image
        dicom = pydicom.dcmread(image_path)
        image = dicom.pixel_array.astype(np.float32)

        # Normalize the image to [0, 1]
        image = (image - image.min()) / (image.max() - image.min())

        # Load NIfTI mask
        mask = nib.load(mask_path).get_fdata().astype(np.float32)

        # Add channel dimension (grayscale images have 1 channel)
        image = np.expand_dims(image, axis=0)  # Shape: (1, height, width)
        mask = np.expand_dims(mask, axis=0)    # Shape: (1, height, width)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

# Prepare dataset
image_folder = os.path.join("DataPreprocess/data/dicom", "data", "dicom")  # Path to DICOM images
mask_folder = os.path.join("DataPreprocess/data/masks", "data", "masks")   # Path to generated masks
dataset = MedicalDataset(image_folder, mask_folder)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize the U-Net++ model for grayscale images (input_channels=1)
model = UNetPlusPlus(num_classes=1, input_channels=1, deep_supervision=False)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss for binary segmentation
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    epoch_loss = 0

    for images, masks in dataloader:
        # Forward pass
        outputs = model(images)  # Correct usage: model(input) calls the forward method
        loss = criterion(outputs, masks)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

# Save the trained model
os.makedirs(os.path.join("..", "models"), exist_ok=True)
torch.save(model.state_dict(), os.path.join("..", "models", "unetpp_grayscale.pth"))
print("Training complete. Model saved.")