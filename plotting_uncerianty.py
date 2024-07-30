import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
file_path = './experiments/Hopper-v4/ade/seed_0/stored_data/variance_of_49976th_episode.npy'
data = np.load(file_path)

# Check the shape and type of the data
print(f'Data shape: {data.shape}')
print(f'Data type: {data.dtype}')

# Moving average function
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Smooth the data
window_size = 15
smoothed_data = moving_average(data, window_size)

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(range(window_size - 1, len(data)), smoothed_data, label='Smoothed Data', color='red')
plt.title('Plot of .npy Data')
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.grid(True)
plt.show()
