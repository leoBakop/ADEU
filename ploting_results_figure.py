import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_images_from_folder(folder_path, save_path):
    l_png_files = []
    # List all files in the given folder
    files = os.listdir(folder_path)
    
    # Filter out only .png files
    png_files = [file for file in files if file.endswith('.png')]
    # Sort the files to ensure consistent order
    png_files.sort()
    for i in range(int(len(png_files)/2)):
        l_png_files.append(png_files[i])
        l_png_files.append(png_files[i+3])

    png_files = l_png_files    
    # Create a 3x2 subplot (3 rows, 2 columns)
    fig, axes = plt.subplots(3, 2, figsize=(50, 50))  # Adjusted figsize for larger subplots
    
    # Loop over the axes and the png files together
    for ax, png_file in zip(axes.flatten(), png_files):
        # Load the image
        img = mpimg.imread(os.path.join(folder_path, png_file))
        # Plot the image on the current axis
        ax.imshow(img)
        # Remove the axis
        ax.axis('off')
        # Optionally, set the title as the filename
        # ax.set_title(png_file)

    # Adjust layout to reduce space between columns and prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 1])
    
    # Add a main title to the figure
    #plt.suptitle('Results in all the Tested Domains', fontsize=24)

    # Save the figure
    plt.savefig(save_path)

    # Show the plot
    plt.show()

# Specify the folder path containing the images and the path to save the plot
folder_path = './huge_figure'
save_path = f'{folder_path}/full_results.png'

# Plot images and save the figure
plot_images_from_folder(folder_path, save_path)
