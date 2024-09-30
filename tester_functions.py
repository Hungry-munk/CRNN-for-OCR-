from configs import Configs
import pathlib as pl
import matplotlib.pyplot as plt

# A function written to display a batch
#I used it to ensure batch generator was working alongside the augmentor model
def inpute_batch_displayer(batch):

    num_images = len(batch)
    cols = 5  # Adjust number of columns to fit your needs
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    axes = axes.flatten()  # Flatten to easily index axes

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(batch[i], cmap='gray')
        ax.axis('off')
        # Add index label on top of each image
        ax.text(0.5, 1.05, str(i), ha='center', va='center', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    # Hide any empty subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def display_single_image(image):
    """
    Displays a single image.

    Args:
        image (ndarray): The input image.
    """

    # Create a figure with a single axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Display the image
    ax.imshow(image, cmap='gray' if len(image.shape) == 2 else None)

    # Remove axis ticks
    ax.axis('off')

    # Tight layout to remove padding
    plt.tight_layout()

    # Display the plot
    plt.show()