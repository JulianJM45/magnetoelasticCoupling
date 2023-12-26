import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from h_res import allHres



# input_filepath = r'C:\Users\Julian\Documents\BA\Field_and_Angle_Sweep#3\M06\ResFields_Rayleigh.txt'
input_filepath = './saw_sw_simulationYK/ResFields_Rayleigh.txt'


# initial guesses
t = 1.0 * 100e-9
k =  0.35 * 13.0081641e6
A = 3.65e-12
g = 2.0269
mue0Hani = -3.01e-5 
mue0Ms =  0.1838 


# Read the data from the .txt file
data = np.loadtxt(input_filepath, dtype=float, skiprows=1, encoding='latin-1')

Angles = np.radians(data[:, 0])
Fields = data[:, 1]*0.001

fitparams = [t, k, A, g, mue0Hani, mue0Ms]

# Define bounds as a percentage of the initial values
bounds_factor = 0.20  # 10 percent
bounds_factor2 = 5
lower_bounds = [param - np.abs(param) * bounds_factor for param in fitparams]
upper_bounds = [param + np.abs(param) * bounds_factor for param in fitparams]
#set high bounds for mue0Hani
for i in [1, 4]:
    lower_bounds[i] = fitparams[i] - np.abs(fitparams[i]) * bounds_factor2
    upper_bounds[i] = fitparams[i] + np.abs(fitparams[i]) * bounds_factor2


bounds = (lower_bounds, upper_bounds)

# Perform the curve fit with bounds
fitparams, covariance = curve_fit(allHres, Angles, Fields, p0=fitparams, bounds=bounds, maxfev=50000)


# Extract the fitted parameters
t_fit, k_fit, A_fit, g_fit, mue0Hani_fit, mue0Ms_fit = fitparams

# Print the fitted parameters
print(f'Fitted t: {t_fit}')
print(f'Fitted k: {k_fit}')
print(f'Fitted A: {A_fit}')
print(f'Fitted g: {g_fit}')
print(f'Fitted mue0Hani: {mue0Hani_fit}')
print(f'Fitted mue0Ms: {mue0Ms_fit}')




# Generate fitted curve
Fields_fit = allHres(Angles, t_fit, k_fit, A_fit, g_fit, mue0Hani_fit, mue0Ms_fit)
r2_resonance = r2_score(Fields, Fields_fit)

print(f'r2 score: {r2_resonance}')


Fields_fit = allHres(Angles, t, k, A, g, mue0Hani, mue0Ms)

# print (Fields_fit)

# Plot the original data
plt.scatter(Fields, np.rad2deg(Angles), label='Original Data')

# Plot the fitted curve
plt.plot(Fields_fit, np.rad2deg(Angles), label='Fitted Curve', color='red')

# Add labels and legend
plt.xlabel('Angles')
plt.ylabel('Fields')
plt.legend()

# Show the plot
plt.show()