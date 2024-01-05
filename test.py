
import matplotlib.pyplot as plt
import numpy as np

# Your 2D dictionary
data = {
    1: {1: 10, 3: 20, 7: 30},
    2: {2: 40, 4: 50, 5: 60},
    3: {1: 70, 8: 80, 11: 90}
}

# Extract x and y values
x = sorted(set(k for k in data.keys()))
y = sorted(set(k for d in data.values() for k in d.keys()))

# Create 2D array from dictionary
values = np.array([[data[i][j] for j in y] for i in x])

# Plot using pcolormesh
plt.pcolormesh(x, y, values, shading='auto')
plt.colorbar(label='Values')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('2D Dictionary Plot')
plt.show()