import numpy as np
import matplotlib.pyplot as plt

# Your data, replace this with your actual data
data = np.array([[1, 2, np.nan, 4],
                 [5, np.nan, 7, 8],
                 [9, 10, 11, 12]])

# Create an array of x values for each column
x_values = np.arange(data.shape[1])

# Create a meshgrid for x and y values
x, y = np.meshgrid(x_values, np.arange(data.shape[0]))

# Plot using pcolormesh, ignoring NaN values
plt.pcolormesh(x, y, data, edgecolors='k', linewidth=2, cmap='viridis')

# Set labels and show the plot
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.colorbar(label='Values')
plt.show()
