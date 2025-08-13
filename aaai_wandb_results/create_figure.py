from PIL import Image

path = 'aaai_wandb_results/'

# List of your 6 image paths
image_paths = [
    path + "hopper_cumulative_rewards_plot.png",
    path + "walker_cumulative_rewards_plot.png",
    path + "swimmer_cumulative_rewards_plot.png",
    path + "ant_cumulative_rewards_plot.png",
    path + "humanoid_cumulative_rewards_plot.png",
    path + "humanoidStandup_cumulative_rewards_plot.png"
]

# Load images
images = [Image.open(p) for p in image_paths]

# Assume all images are the same size
img_width, img_height = images[0].size

# Set grid dimensions: 2 columns, 3 rows
cols = 2
rows = 3
grid_width = img_width * cols
grid_height = img_height * rows

# Create a new blank image for the grid
grid_img = Image.new('RGB', (grid_width, grid_height))

# Paste each image in its proper position
for index, img in enumerate(images):
    x = (index % cols) * img_width   # column
    y = (index // cols) * img_height # row
    grid_img.paste(img, (x, y))

# Save or show the resulting grid image
grid_img.save(path + "full_figure.png")
grid_img.show()
