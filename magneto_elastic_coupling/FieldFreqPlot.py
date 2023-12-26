import os
import numpy as np
import matplotlib.pyplot as plt


# Define the angle you want to measure
alpha = 0

# Define file paths
input_filepath = r'C:\Users\Julian\Documents\BA\Field_Sweep#3\Sezawa_3.53GHz\Sezawa3.55_-45deg.txt'
output_filepath = r'C:\Users\Julian\Documents\BA\Field_Sweep#3\Sezawa_3.53GHz\Sezawa3.55_-45deg.png'
# Read the data from the .txt file
data = np.loadtxt(input_filepath, dtype=float, skiprows=1)

# Extract the angles
Angle_column = data[:, 0]
data = data[(Angle_column >= -90) & (Angle_column <= 90)]

#Extract columns
setField_column = data[:, 1]
Angle_column = data[:, 0]
Field_column = 1000 * (data[:, 2] + data[:, 3]) / 2
S12_column = data[:, 5]

print(S12_column)

# Calculate average S_12 of 3 min/max field elemnts for each angle
kth = 3
average_s12 = {}
unique_angles = np.unique(Angle_column)
for angle in unique_angles: 
    mask = Angle_column == angle
    smallest_fields = np.partition(Field_column[mask], kth-1)[:kth]
    mask2 = np.isin(Field_column[mask], smallest_fields)
    avg_s12_smallest = np.mean(S12_column[mask][mask2])
    largest_fields = np.partition(Field_column[mask], -kth)[-kth:]
    mask2 = np.isin(Field_column[mask], largest_fields)
    avg_s12_largest = np.mean(S12_column[mask][mask2])
    average_s12[angle] = (avg_s12_smallest+avg_s12_largest)/2

# Calculate the Delta S_12 column
deltaS12_column = np.zeros_like(S12_column)
for angle in unique_angles:
    mask = Angle_column == angle
    delta_S12 = S12_column[mask] - average_s12[angle]
    # Handle invalid or missing values (e.g., NaN)
    delta_S12[np.isnan(delta_S12)] = 0  # Replace NaN with 0 or another suitable value
    deltaS12_column[mask] = delta_S12


# Filter data for the angle alpha (in your case, alpha = 50)
angle_filter = Angle_column == alpha
filtered_field = Field_column[angle_filter]
filtered_S12 = deltaS12_column[angle_filter]


# Create a plot
# Set the desired figure size (width_cm, height_cm) in centimeters
width_cm = 16  # Adjust width as needed in cm
height_cm = 11  # Adjust height as needed in cm
fontsize = 16

# Convert centimeters to inches
width_in = width_cm / 2.54
height_in = height_cm / 2.54


cmap = plt.get_cmap('hot')
# Create the figure with the specified size
#plt.figure(figsize=(width_in, height_in))
#plt.scatter(filtered_field, filtered_S12, c=filtered_S12, cmap=cmap)
plt.xlabel('Field $\mu_0H$ (mT)', fontsize=fontsize)
plt.ylabel('$\Delta S_{12}$ (dB)', fontsize=fontsize)
plt.title('Rayleigh $\Delta S_{12}$ for angle $45°$', fontsize=fontsize)
plt.xticks(fontsize=fontsize)  
plt.yticks(fontsize=fontsize)  

plt.minorticks_on()

# Connect the dots with lines
plt.plot(filtered_field, filtered_S12, 'black', alpha=0.5, label='Connected Line')

# Save the plot to a file
plt.savefig(output_filepath, bbox_inches='tight')

# Show the plot (optional)
plt.show()

