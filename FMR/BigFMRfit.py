import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


# Specify the root directory where your .dat files are located
root_dir = r'C:\Users\Julian\Documents\BA\FMR#3\singleAngle'

#initial guesses for fit:
g1 = 2
Hani1 = 0.043
Ms1 = 0.057
alpha1 = 0.00078
dHinhomo1 = 0.0035

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

# Initialize empty lists to store 'f' and 'H' columns
all_f_columns = []
all_H_columns = []
all_dHres_columns = []
all_dHresErr_columns = []
angle_list = []
# Iterate through directories with names like Angle_X.0(FTF)
for angle_dir in os.listdir(root_dir):
    if angle_dir.startswith("Angle_") and os.path.isdir(os.path.join(root_dir, angle_dir)):
        angle_list.append(angle_dir)
        # Construct the full path to the .dat file in the current directory
        file_path = os.path.join(root_dir, angle_dir, '1. FMR-Susceptibility Fit', 'Resonance Fit.dat')

        # Check if the file exists
        if os.path.isfile(file_path):
            # Read the data into a pandas DataFrame
            data = pd.read_csv(file_path, delimiter='\t', skipinitialspace=True, names=column_names, skiprows=2)

            # Extract the 'f (GHz)' and 'Hres1' columns and append to the lists
            all_f_columns.append(data['f (GHz)'] * 1e9)
            all_H_columns.append(-data['Hres1'])
            all_dHres_columns.append(data['dH1'])
            all_dHresErr_columns.append(data['dH err1'])
        
# Initialize data and parameters
angle_count = len(angle_list)
angle_values = [float(angle.split('_')[1].replace('(FTF)', '')) for angle in angle_list]
# print(angle_values)
LEN = len(all_f_columns[0])
all_H_array = np.concatenate(all_H_columns)
all_f_array = np.concatenate(all_f_columns)
all_dHres_array = np.concatenate(all_dHres_columns)
all_dHresErr_array = np.concatenate(all_dHresErr_columns)

# invert_kittel_equation
def invert_kittel_equation(f, g, Ms, Hani):
    # Constants
    hbar = 1.054571817e-34
    mu_B = 9.2740100783e-24
    pi = np.pi

    Hani = Hani / mu_B
    Ms = Ms / mu_B
    # Coefficients for the quadratic equation
    a = 1
    b = 2 * Hani + Ms
    c = Hani**2 + Hani * Ms - (4 * pi**2 * hbar**2) / (g**2 * mu_B**2 * f**2)

    # Calculate the discriminant
    discriminant = b**2 - 4 * a * c

    # Check if the discriminant is non-negative for real solutions
    if discriminant >= 0:
        # Calculate the two possible solutions
        root1 = mu_B*(-b + np.sqrt(discriminant)) / (2 * a)
        root2 = mu_B*(-b - np.sqrt(discriminant)) / (2 * a)
        
        return root1
    else:
        # No real solutions
        print('negative Discriminante')
        return None
    

# Define the Kittel equation
def kittel_equation(H, g, Ms, Hani):
    sqrt_term = np.sqrt((H + Hani) * (H + Hani + Ms))
    result = ((g * 9.2740100783e-24 / (2 * np.pi * 1.054571817e-34)) * sqrt_term)
    return result

def kittel_equationALL(H_fields, g, Ms, *Hani_values):
    result = []
    Hani_values = np.array(Hani_values)
    H_fields = np.array(H_fields)
    for i, Hani in enumerate(Hani_values):
        H_subset = H_fields[i * LEN: (i + 1) * LEN]
        for H in H_subset:
            sqrt_term = np.sqrt((H + Hani) * (H + Hani + Ms))
            result.append((g * 9.2740100783e-24 / (2 * np.pi * 1.054571817e-34)) * sqrt_term)
    return result

# Define linewidth equation
def linewidth_equation(f, alpha, dHinhomo):
    result = ((alpha * 2 * np.pi * 1.054571817e-34)/(g_fit * 9.2740100783e-24) * f + dHinhomo)
    return result

def linewidth_equationALL(f_fields, alpha, *dHinhomo_values):
    result = []
    dHinhomo_values = np.array(dHinhomo_values)
    f_fields = np.array(f_fields)
    for i, dHinhomo in enumerate(dHinhomo_values):
        f_subset = f_fields[i * LEN: (i + 1) * LEN]
        for f in f_subset:
            result.append((alpha * 4 * np.pi * 1.054571817e-34)/(g_fit * 9.2740100783e-24) * f + dHinhomo)
    return result


def exportData(angle_values, Hani_fit_array):
    new_data = np.column_stack((angle_values, Hani_fit_array))
    header = np.array(["Angle in °", "Hani_fit in T"])
    output_filepath = os.path.join(root_dir, f'HaniFits.txt')
    np.savetxt(output_filepath, new_data, header='\t'.join(header), comments='', delimiter='\t', fmt='%.6e')


# Initial guess for the parameters (you may need to adjust these)
initial_guess_kittel = np.hstack([g1, Ms1, np.full(angle_count, Hani1)])
initial_guess_linewidth = np.hstack([alpha1, np.full(angle_count, dHinhomo1)])


# Perform the kittel fit
params, cost = curve_fit(kittel_equationALL, all_H_array, all_f_array, initial_guess_kittel)
g_fit, Ms_fit, *Hani_fit_array = params

#perform the linewidth fit
params2, covariance2 = curve_fit(linewidth_equationALL, all_f_array, all_dHres_array, p0=initial_guess_linewidth)
alpha_fit, *dHinhomo_fit_array = params2


exportData(angle_values, Hani_fit_array)

# Print the fitted parameters
print(f"Fitted g: {g_fit}")
# print(f"Fitted Hani: {Hani_fit_array}")
print(f"Fitted meanHani: {np.mean(Hani_fit_array)}")
print(f"Fitted Ms: {Ms_fit}")
print(f"Fitted alpha: {alpha_fit}")
# print(f"Fitted dHinhomo: {dHinhomo_fit_array}")
print(f"Fitted meandHinhomo: {np.mean(dHinhomo_fit_array)}")


#get res freq for H=65mT
# H0 = 0.65
# f1 = kittel_equation(H0, g_fit, Ms_fit, np.mean(Hani_fit_array))
f1 = 3.53e9
H1 = invert_kittel_equation(f1, g_fit, Ms_fit, np.mean(Hani_fit_array))
print(f'Resonace Field for f = {f1}Hz: {H1}T')

# Plot the data and the fitted curves
plt.figure(figsize=(12, 6))

# Generate fitted curve and plot kittel
plt.subplot(1, 3, 1)
plt.scatter(all_H_array, all_f_array, label='Data')
H_fit = np.linspace(0, max(all_H_array), LEN)
for i, Hani_fit in enumerate(Hani_fit_array):
    f_fit = kittel_equation(H_fit, g_fit, Ms_fit, Hani_fit)
    # r2_resonance = r2_score(all_f_array, f_fit)
    plt.plot(H_fit, f_fit)#, label=f'Fit (Angle={angle_values[i]:.3f}°)', alpha=0.7)    
plt.xlabel('Effective Magnetic Field (H)')
plt.ylabel('Resonance Frequency (f)')
plt.legend()

# Generate fitted curve and plot linewidth
plt.subplot(1, 3, 2)
plt.scatter(all_f_array, all_dHres_array, label='Data')
plt.errorbar(all_f_array, all_dHres_array, yerr=all_dHresErr_array, fmt='o', label='Data with Error')
f_fit2 = np.linspace(min(all_f_array), max(all_f_array), LEN)
for i, dHinhomo_fit in enumerate(dHinhomo_fit_array):
    dHres_fit = linewidth_equation(f_fit2, alpha_fit, dHinhomo_fit)
    plt.plot(f_fit2, dHres_fit)#, label=f'Fit (Angle={angle_values[i]:.3f}°)', alpha=0.7)
plt.plot(f_fit2, dHres_fit, label='Fit', color='red')
plt.xlabel('Resonance Frequency (f)')
plt.ylabel('Linewidth (dHres)')
plt.legend()

# Plot Hani_fit_array over angle_values
plt.subplot(1, 3, 3)
plt.plot(angle_values, Hani_fit_array, marker='o', linestyle='', color='green')
plt.xlabel('Angle (degrees)')
plt.ylabel('Fitted Hani')
plt.title('Fitted Hani over Angle')
plt.grid(True)

#show plot
plt.tight_layout()
plt.show()




