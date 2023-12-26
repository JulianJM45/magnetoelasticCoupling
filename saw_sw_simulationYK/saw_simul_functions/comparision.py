import numpy as np
import os
from Plot import plot
import matplotlib.pyplot as plt

input_filepath_Mathematica = r'C:\Users\Julian\Documents\Wolfram Mathematica\Pim_2.43Ghz.txt'
input_filepath_Matlab = r'C:\Users\Julian\Documents\MATLAB\SimulationYannik\P_abs.txt'
output_folder = r'C:\Users\Julian\Pictures\ComparisionSimulation_xx'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Load Mathematica Data
data_math = np.loadtxt(input_filepath_Mathematica, dtype=float)
Fields = np.unique(data_math[:, 0])
Angles = np.unique(data_math[:, 1])
S_12_column_math = data_math[:, 2]

# Reshape S_12_column_math to match the dimensions of Field_column_math and Angle_column_math
num_rows_math = len(Angles)
num_cols_math = len(Fields)
P_abs_math = -S_12_column_math.reshape(num_rows_math, num_cols_math)

# Load Matlab Data
data_matlab = np.loadtxt(input_filepath_Matlab, delimiter=',')  # Specify delimiter if needed
P_abs_matlab = np.array(data_matlab)


# Normalize
# Min-max scaling
def min_max_scaling(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    scaled_matrix = (matrix - min_val) / (max_val - min_val)
    return scaled_matrix

# Apply min-max scaling
P_abs_math_scaled = min_max_scaling(P_abs_math)
P_abs_matlab_scaled = min_max_scaling(P_abs_matlab)


# Specify the angle to plot
angles_to_plot = [-90, -80, -70, -62, -50, -40, -32, -20, -10, 0, 10, 20, 32, 40, 50, 62, 70, 80, 90]
lenPlotAngles = len(angles_to_plot)

# Plot the results
plt.figure(figsize=(12, 5))

plt.xlabel('Fields')
plt.ylabel('Scaled P_abs')
# plt.legend()

# # Subplot for Mathematica
# plt.subplot(1, 2, 1)

for angle_to_plot in angles_to_plot:
    # Find the index where the angle is equal to the angle_to_plot
    angle_index_to_plot = np.where(Angles == angle_to_plot)[0]

    # Extract the corresponding data for the selected angle
    P_abs_math_selected_angle = P_abs_math_scaled[angle_index_to_plot, :].flatten()
    P_abs_matlab_selected_angle = P_abs_matlab_scaled[angle_index_to_plot, :].flatten()

    # Plot for Mathematica and MATLAB
    plt.plot(Fields, P_abs_math_selected_angle, label=f'Mathematica', )
    plt.plot(Fields, P_abs_matlab_selected_angle, label=f'Matlab')

    plt.title(f'P_abs Scaled {angle_to_plot}°')

    plt.legend(loc='best')

    # plt.show()
    filepath_output = os.path.join(output_folder, f'SimulatedColorMap2.43GHz_{angle_to_plot}°.png')

    plt.savefig(filepath_output, dpi=300, bbox_inches='tight')

    plt.clf()







# plt.tight_layout()
# plt.show()

filepath_output = os.path.join(output_folder, f'SimulatedColorMap2.43GHz_Mathematica.png')
plot(Angles, Fields, P_abs_math_scaled, filepath_output)
filepath_output = os.path.join(output_folder, f'SimulatedColorMap2.43GHz_Matlab.png')
plot(Angles, Fields, P_abs_matlab_scaled, filepath_output)