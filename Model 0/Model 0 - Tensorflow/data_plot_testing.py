from matplotlib import pyplot as plt

from data_loader import images_array, masks_array


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