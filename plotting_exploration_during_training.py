import os
from PIL import Image
import matplotlib.pyplot as plt

# Function to load images from a folder and plot them side by side
def plot_images_side_by_side(folder_path):
    # List all files in the directory
    files = os.listdir(folder_path)
    
    # Filter out the PNG files
    png_files = [file for file in files if file.endswith('.png')]
    
    # Ensure there are exactly two PNG files
    if len(png_files) != 2:
        print("Error: There should be exactly two PNG files in the folder.")
        return
    
    # Load the images
    img1 = Image.open(os.path.join(folder_path, png_files[1]))
    img2 = Image.open(os.path.join(folder_path, png_files[0]))
    
    # Create a figure and axis to plot the images
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    
    # Plot the first image
    axes[0].imshow(img1)
    axes[0].axis('off')  # Turn off axis labels
    #axes[0].set_title(png_files[0])
    
    # Plot the second image
    axes[1].imshow(img2)
    axes[1].axis('off')  # Turn off axis labels
    #axes[1].set_title(png_files[1])
    
    # Show the plot
    plt.tight_layout()
    #plt.savefig('./figures/exploration_rate_combined.png')
    plt.show()

# Example usage
folder_path = './figures'  # Replace with your folder path
plot_images_side_by_side(folder_path)
