import os
import numpy as np
import matplotlib.pyplot as plt

input_filepath = r'C:\Users\Julian\Documents\Wolfram Mathematica\Pim_2.43Ghz.txt'
output_filepath = r'C:\Users\Julian\Pictures\SimulatedColorMap2.43GHz.png'

data = np.loadtxt(input_filepath, dtype=float)


Field_column = data[:, 0]
Angle_column = data[:, 1]
S_12_column = data[:, 2]

# S_12_column = np.exp(S_12_column/10**17)


'''
# Calculate average S_12 of 3 min/max field elemnts for each angle
kth = 3
average_s12 = {}
unique_angles = np.unique(Angle_column)
for angle in unique_angles: 
    mask = Angle_column == angle
    smallest_fields = np.partition(Field_column[mask], kth-1)[:kth]
    mask2 = np.isin(Field_column[mask], smallest_fields)
    avg_s12_smallest = np.mean(S_12_column[mask][mask2])
    largest_fields = np.partition(Field_column[mask], -kth)[-kth:]
    mask2 = np.isin(Field_column[mask], largest_fields)
    avg_s12_largest = np.mean(S_12_column[mask][mask2])
    average_s12[angle] = (avg_s12_smallest+avg_s12_largest)/2

# Calculate the Delta S_12 column
deltaS12_column = np.zeros_like(S_12_column)
for angle in unique_angles:
    mask = Angle_column == angle
    delta_S12 = S_12_column[mask] - average_s12[angle]
    # Handle invalid or missing values (e.g., NaN)
    delta_S12[np.isnan(delta_S12)] = 0  # Replace NaN with 0 or another suitable value
    deltaS12_column[mask] = delta_S12

'''

# Reshape S_12_colomn to match the dimensions of Field_column and Angle_column
num_rows = len(np.unique(Angle_column))
num_cols = len(np.unique(Field_column))
S_12_matrix = S_12_column.reshape(num_rows, num_cols)

im = plt.imshow(S_12_matrix, extent=[min(Field_column),max(Field_column),min(Angle_column),max(Angle_column)], cmap='hot_r', interpolation='nearest', aspect='auto')

#labels 
fontsize = 14
plt.minorticks_on()
cbar = plt.colorbar(im)
cbar.ax.set_title('$\Delta$$S_{21}$ (dB)', fontsize=fontsize)
plt.xlabel('Field $\mu_0H$ (mT)', fontsize=fontsize)
plt.ylabel('$θ_H$ (°)', fontsize=fontsize)
plt.title('Simulated $\Delta$$S_{12}$ for Rayleigh mode', fontsize=fontsize)
plt.xticks(fontsize=fontsize)  
plt.yticks(fontsize=fontsize)

#plt.show()
plt.savefig(output_filepath, dpi=300, bbox_inches='tight')

plt.show()