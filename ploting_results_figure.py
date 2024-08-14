import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_images_from_folder(folder_path, save_path):
    # List all files in the given folder
    files = os.listdir(folder_path)
    
    # Filter out only .png files
    png_files = [file for file in files if file.endswith('.png')]
    
    # Sort the files to ensure consistent order
    png_files.sort()

    # Create a 2x4 subplot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Loop over the axes and the png files together
    for ax, png_file in zip(axes.flatten(), png_files):
        # Load the image
        img = mpimg.imread(os.path.join(folder_path, png_file))
        # Plot the image on the current axis
        ax.imshow(img)
        # Remove the axis
        ax.axis('off')
        # Optionally, set the title as the filename
        #ax.set_title(png_file)

    # Add a main title to the figure
    plt.suptitle('Results in all the Tested Domains', fontsize=16)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the figure
    plt.savefig(save_path)

    # Show the plot
    plt.show()

# Specify the folder path containing the images and the path to save the plot

folder_path = './huge_figure'
save_path = f'{folder_path}/full_results.png'

# Plot images and save the figure
plot_images_from_folder(folder_path, save_path)
