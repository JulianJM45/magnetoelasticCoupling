import os
import numpy as np
import matplotlib.pyplot as plt
import cmath

# Define file paths
input_folder= r'C:\Users\Julian\Documents\MATLAB\960nm thickness estimated from quartz'
input_filenameRayleigh = 'Displacements2.45GHz960nmZnO103nmYIG200nmDeadlayerIDT_FLLLS_c-axisangle0degrees.txt'
filepath_output = r'C:\Users\Julian\Documents'


input_filepath = os.path.join(input_folder, input_filenameRayleigh)

# Define parameters
k = 5.5669e06 # in 1/m
c11 = 209.7e09 #in Pa
c66 = 44.3e09 #in Pa
rho_YIG = 5.17e03 #in kg/m^3
f= 2.43e09 #in Hz

# calculate chi
v_L = np.sqrt(c11/rho_YIG)
v_T = np.sqrt(c66/rho_YIG)
v = 2*np.pi*f/k

chi1 = np.sqrt(1-v**2/v_L**2)
chi3 = np.sqrt(1-v**2/v_T**2)


# Read the data from the .txt file
data = np.loadtxt(input_filepath, dtype=float, delimiter=',')

# Find the indices of rows where the "z" value is within the range [1, 99]
start_row = np.where(data[:, 0] == 1)[0][0]  # Find the first occurrence of "z" = 1
end_row = np.where(data[:, 0] == 99)[0][0] + 1  # Find the first occurrence of "z" = 99 and add 1 to include it

# Slice the data to include rows within the specified range of "z" values
selected_data = data[start_row:end_row, :]

# Calculate the mean value for each column
mean_values = np.mean(selected_data, axis=0)

ux_re = mean_values[1]
uy_re = mean_values[2]
uz_re = mean_values[3]
ux_im = mean_values[5]*1j
uy_im = mean_values[6]*1j
uz_im = mean_values[7]*1j

# calculate Strain components
duxdx = -1j*k*(ux_re+ux_im)
duxdz = -k*(chi1*ux_re+chi3*ux_im)
duzdx = -1j*k*(uz_re+uz_im)
duzdz = -k*(chi3*uz_re+chi1*uz_im)

epsxx = duxdx
epsxz = 0.5*(duxdz+duzdx)
epszz = duzdz
epszz = 0


# nominierung
maxvalue = max(abs(epsxx.real), abs(epsxx.imag), abs(epsxz.real), abs(epsxz.imag), abs(epszz.real), abs(epszz.imag))




print('epsilon xx:',epsxx/maxvalue, 'epsilon xz:', epsxz/maxvalue, 'epsilon zz:', epszz/maxvalue)