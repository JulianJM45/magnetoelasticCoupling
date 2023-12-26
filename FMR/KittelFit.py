import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Specify the path to your .dat file
file_path = r'C:\Users\Julian\Documents\BA\FMR#3\singleAngle\Angle_72.0(FTF)\1. FMR-Susceptibility Fit\Resonance Fit.dat'

# Define column names based on your data
column_names = [
    'f (GHz)', 'Phi (deg)', 'Theta (deg)', 'Power (dBm)',
    'Residue', 'SSE', 'R-Square', 'RMSE',
    'offset RE', 'offset RE err', 'slope RE', 'slope RE err',
    'offset IM', 'offset IM err', 'slope IM', 'slope IM err',
    'Hres1', 'Hres err1', 'dH1', 'dH err1',
    'A1', 'A err1', 'A*chi(H_res) 1', 'A*chi(H_res) err1',
    'Phi1', 'Phi err1'
]

# Read the data into a pandas DataFrame
data = pd.read_csv(file_path, delimiter='\t', skipinitialspace=True, names=column_names, skiprows=1)
# data = data.iloc[1:]
# Extract the 'f (GHz)' column
f_column = data['f (GHz)']*1e9
H_column = -data['Hres1']
dHres_column = data['dH1']
dHresErr_column = data['dH err1']

print(f_column)

# Define the Kittel equation
def kittel_equation(H, g, Hani, Ms):
    return (g * 9.2740100783e-24 / (2 * np.pi * 1.054571817e-34)) * np.sqrt((H + Hani) * (H + Hani + Ms))

# Initial guess for the parameters (you may need to adjust these)
initial_guess = [2.0, 0.03, 0.6]

# Perform the curve fit
params, covariance = curve_fit(kittel_equation, H_column, f_column, p0=initial_guess)

# Extract the fitted parameters
g_fit, Hani_fit, Ms_fit = params

# Generate fitted curve
H_fit = np.linspace(min(H_column), max(H_column), len(f_column))
f_fit = kittel_equation(H_fit, g_fit, Hani_fit, Ms_fit)
r2_resonance = r2_score(f_column, f_fit)

# Define linewidth equation
def linewidth_equation(f, alpha, dHinhomo):
    return ((alpha * 2 * np.pi * 1.054571817e-34)/(g_fit * 9.2740100783e-24) * f + dHinhomo)

initial_guess2 = [0.000199352, 0.000562977]

params2, covariance2 = curve_fit(linewidth_equation, f_column, dHres_column, p0=initial_guess2)

alpha_fit, dHinhomo_fit = params2

f_fit2 = np.linspace(min(f_column), max(f_column), len(dHres_column))
dHres_fit = linewidth_equation(f_fit2, alpha_fit, dHinhomo_fit)
r2_linewidth = r2_score(dHres_column, dHres_fit)


# Print the fitted parameters
print(f"Fitted g: {g_fit}")
print(f"Fitted Hani: {Hani_fit}")
print(f"Fitted Ms: {Ms_fit}")
print(f"Fitted alpha: {alpha_fit}")
print(f"Fitted dHinhomo: {dHinhomo_fit}")
print(f"R^2 for resonance frequency fit: {r2_resonance}")
print(f"R^2 for linewidth fit: {r2_linewidth}")

# Plot the data and the fitted curves
plt.figure(figsize=(12, 6))

# Plot for resonance frequency fit
plt.subplot(1, 2, 1)
plt.scatter(H_column, f_column, label='Data')
plt.plot(H_fit, f_fit, label='Fit', color='red')
plt.xlabel('Effective Magnetic Field (H)')
plt.ylabel('Resonance Frequency (f)')
plt.legend()

# Add annotations for fitted parameters and R^2 for resonance frequency fit
plt.text(0.05, 0.75, f'Fitted g: {g_fit}\nFitted Hani: {Hani_fit}\nFitted Ms: {Ms_fit}\nR^2: {r2_resonance}',
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

# Plot for linewidth fit
plt.subplot(1, 2, 2)
plt.scatter(f_column, dHres_column, label='Data')
plt.errorbar(f_column, dHres_column, yerr=dHresErr_column, fmt='o', label='Data with Error')
plt.plot(f_fit2, dHres_fit, label='Fit', color='red')
plt.xlabel('Resonance Frequency (f)')
plt.ylabel('Linewidth (dHres)')
plt.legend()

# Add annotations for fitted parameters and R^2 for linewidth fit
plt.text(0.05, 0.75, f'Fitted alpha: {alpha_fit}\nFitted dHinhomo: {dHinhomo_fit}\nR^2: {r2_linewidth}',
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

