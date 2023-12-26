import os
import numpy as np
import matplotlib.pyplot as plt

input_filepathReal = r'C:\Users\Julian\Documents\python\TimeSpaceReal.txt'
input_filepathImag = r'C:\Users\Julian\Documents\python\TimeSpaceImag.txt'

dataReal = np.loadtxt(input_filepathReal, dtype=float, skiprows=1)
dataImag = np.loadtxt(input_filepathImag, dtype=float, skiprows=1)

time = dataReal[:, 0]
real = dataReal[:, 1]
imag = dataImag[:, 1]
amplitude = real


Bandwith = 80000
B = 1000*1/Bandwith

time = time*B

y_min = -10* 10**-7
y_max = 11* 10**-7

x_min = 0*B
x_max = 5000*B

# Set the desired figure size (width_cm, height_cm) in centimeters
width_cm = 16  # Adjust width as needed in cm
height_cm = 10  # Adjust height as needed in cm

# Convert centimeters to inches
width_in = width_cm / 2.54
height_in = height_cm / 2.54

# Create the figure with the specified size
fig = plt.figure(figsize=(width_in, height_in))
ax = fig.add_subplot(111)

plt.plot(time, amplitude, linestyle='-', marker='.', markersize=1, color='black')

# Add labels and a legend
plt.xlabel('time (ms)')
plt.ylabel('Amplitude real $S_{12}$')
plt.title('time space')
plt.minorticks_on()

# Define the x-values for the shaded area
left = 3033*B
right = 3210*B

# Fill the area between left and right with transparent blue
ax.fill_between([left, right], y_min, y_max, color='blue', alpha=0.3)

# Define the x-values for the shaded area
left = 3411*B
right = 3663*B

# Fill the area between left and right with transparent blue
ax.fill_between([left, right], y_min, y_max, color='blue', alpha=0.3)

# Set the y-axis limits
plt.ylim(y_min, y_max)
plt.xlim(x_min, x_max)

# Show the plot
plt.show()