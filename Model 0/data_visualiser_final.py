import numpy as np
import matplotlib.pyplot as plt

# Function to display original, ground truth, and predicted masks
def display_predictions(images, true_masks, predicted_masks, num_images=10):
    plt.figure(figsize=(15, num_images * 3))

    for i in range(num_images):
        # Original Image
        plt.subplot(num_images, 3, i * 3 + 1)
        plt.imshow(images[i].reshape(128, 128), cmap='gray')
        plt.title("Original Image")
        plt.axis('off')

        # Ground Truth Mask
        plt.subplot(num_images, 3, i * 3 + 2)
        plt.imshow(true_masks[i].reshape(128, 128), cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis('off')

        # Predicted Mask
        plt.subplot(num_images, 3, i * 3 + 3)
        # Ensure the predicted mask is in the correct shape for visualization
        pred_mask = predicted_masks[i].reshape(128, 128) if len(predicted_masks.shape) == 4 else predicted_masks[i]
        plt.imshow(pred_mask, cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
