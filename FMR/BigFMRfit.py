import pandas as pd
import os
import numpy as np
from my_modules import *
import scipy
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


input_folder = '/home/julian/BA/dataForPython/FMR#3/ResonanceData'
output_folder = '/home/julian/BA/pictures'

LEN = 0

Gfit = 0

#constants
h = scipy.constants.h
mue_B = scipy.constants.e * scipy.constants.hbar /  (2* scipy.constants.m_e)


def main():
    frequency, H, dHres, dHres_err = getData()

    g_fit, Ms_fit, Hani_fit_dict = KittelFit(frequency, H)

    meanHani = np.mean(list(Hani_fit_dict.values()))
    f = 3.53e9
    print(invert_kittel_equation(f, g_fit, Ms_fit, meanHani))

    # exportData(Hani_fit_dict)

    # KittelPlot(g_fit, Ms_fit, Hani_fit_dict)

    # alpha_fit, dHinhomo_fit_dict = LinewidthFit(frequency, dHres)

    # LinewidthPlot(alpha_fit, dHinhomo_fit_dict)

    # dHinhomoPlot(dHinhomo_fit_dict)

    





def LinewidthFit(frequency, dHres):
    angle_count = len(frequency)
    all_f_array = np.concatenate(list(frequency.values()))
    all_dHres_array = np.concatenate(list(dHres.values()))

    #initial guesses for fit:
    alpha1 = 0.00078
    dHinhomo1 = 0.0035
    initial_guess_linewidth = np.hstack([alpha1, np.full(angle_count, dHinhomo1)])

    params, covariance = curve_fit(linewidth_equationALL, all_f_array, all_dHres_array, p0=initial_guess_linewidth)
    alpha_fit, *dHinhomo_fit_array = params
    dHinhomo_fit_dict = dict(zip(frequency.keys(), dHinhomo_fit_array))
 
    print(f"Fitted alpha: {alpha_fit}")
    # # print(f"Fitted dHinhomo: {dHinhomo_fit_array}")
    # print(f"Fitted meandHinhomo: {np.mean(dHinhomo_fit_array)}")
 
    return alpha_fit, dHinhomo_fit_dict


def LinewidthPlot(alpha_fit, dHinhomo_fit_dict):
    colors = get_plot_colors(len(dHinhomo_fit_dict))
    mygraph = Graph()
    for i, key  in enumerate(dHinhomo_fit_dict.keys()): 
        color = colors[i]
        f = np.linspace(0, 25e9, 200)
        dH_res = linewidth_equation(f, alpha_fit, dHinhomo_fit_dict[key])
        mygraph.add_plot(dH_res*1e3, f*1e-9, color=color, label=f'{key}')

    mygraph.plot_Graph(safe=True, legend=False, xlabel='$f$ in GHz', ylabel='$\mu_0 \Delta H_\mathrm{res}$ in mT', name=f'dHresPlot', outputfolder=output_folder)

def dHinhomoPlot(dHinhomo_fit_dict):
    angles = list(dHinhomo_fit_dict.keys())
    dHin = np.array(list(dHinhomo_fit_dict.values()))
    GraphPlot(angles, dHin*1e3, save=True, name="deltaHinhomo", xlabel='Angles in °', ylabel='$\mu_0 \Delta H_\mathrm{inhomo}$ in mT')



    

def KittelPlot(g_fit, Ms_fit, Hani_fit_dict):
    colors = get_plot_colors(len(Hani_fit_dict))
    mygraph = Graph()
    for i, key  in enumerate(Hani_fit_dict.keys()): 
        color = colors[i]
        H_field = np.linspace(0, 1, 200)
        f = kittel_equation(H_field, g_fit, Ms_fit, Hani_fit_dict[key])
        mygraph.add_plot(H_field*1e3, f*1e-9, color=color, label=f'')

    mygraph.plot_Graph(safe=True, legend=False, xlabel='$\mu_0 H$ in mT', ylabel='$f$ in GHz', name=f'KittelPlot', outputfolder=output_folder)


def KittelFit(frequency, H):
    angle_count = len(frequency)
    all_f_array = np.concatenate(list(frequency.values()))
    all_H_array = np.concatenate(list(H.values()))


    #initial guesses for fit:
    g1 = 2
    Hani1 = 0.043
    Ms1 = 0.057

    # Initial guess for the parameters (you may need to adjust these)
    initial_guess_kittel = np.hstack([g1, Ms1, np.full(angle_count, Hani1)])

    # Perform the kittel fit
    params, cost = curve_fit(kittel_equationALL, all_H_array, all_f_array, initial_guess_kittel)
    g_fit, Ms_fit, *Hani_fit_array = params
    Hani_fit_dict = dict(zip(frequency.keys(), Hani_fit_array))

    global Gfit
    Gfit = g_fit

    
    # Print the fitted parameters
    print(f"Fitted g: {g_fit}")
    # print(f"Fitted Hani: {Hani_fit_array}")
    # print(f"Fitted meanHani: {np.mean(Hani_fit_array)}")
    print(f"Fitted Ms: {Ms_fit}")
    
    return g_fit, Ms_fit, Hani_fit_dict





def getData():
    frequency = {}
    H = {}
    dHres = {}
    dHres_err = {}

    column_names = [
        'f (GHz)', 'Phi (deg)', 'Theta (deg)', 'Power (dBm)',
        'Residue', 'SSE', 'R-Square', 'RMSE',
        'offset RE', 'offset RE err', 'slope RE', 'slope RE err',
        'offset IM', 'offset IM err', 'slope IM', 'slope IM err',
        'Hres1', 'Hres err1', 'dH1', 'dH err1',
        'A1', 'A err1', 'A*chi(H_res) 1', 'A*chi(H_res) err1',
        'Phi1', 'Phi err1'
    ]
    angles = []  # To store the angles for sorting
    for angleFile in os.listdir(input_folder):
        angle = int(float(angleFile.split('_')[1].replace('(FTF)', '')))
        angles.append(angle)  # Collect angles for sorting
        file_path = os.path.join(input_folder, angleFile)
        data = pd.read_csv(file_path, delimiter='\t', skipinitialspace=True, names=column_names, skiprows=2)
        frequency[angle] = data['f (GHz)'] * 1e9
        H[angle] = -data['Hres1']
        dHres[angle]  = data['dH1']
        dHres_err[angle] = data['dH err1']

     # Sort the angles in ascending order
    angles.sort()

    # Create sorted dictionaries using the sorted angles
    sorted_frequency = {angle: frequency[angle] for angle in angles}
    sorted_H = {angle: H[angle] for angle in angles}
    sorted_dHres = {angle: dHres[angle] for angle in angles}
    sorted_dHres_err = {angle: dHres_err[angle] for angle in angles}

    # Update the global variable LEN
    global LEN
    LEN = len(sorted_frequency[angles[0]])

    return sorted_frequency, sorted_H, sorted_dHres, sorted_dHres_err


# invert_kittel_equation
def invert_kittel_equation(f, g, Ms, Hani):
    # Coefficients for the quadratic equation
    a = 1
    b = 2 * Hani + Ms
    c = Hani * (Hani + Ms) - ((f * h) / (g * mue_B))**2

    # Calculate the discriminant
    discriminant = b**2 - 4 * a * c

    # Check if the discriminant is non-negative for real solutions
    if discriminant >= 0:
        # Calculate the two possible solutions
        root1 = (-b + np.sqrt(discriminant)) / (2 * a)
        root2 = (-b - np.sqrt(discriminant)) / (2 * a)
        
        return root1
    else:
        # No real solutions
        print('negative Discriminante')
        return None
    

# Define the Kittel equation
def kittel_equation(H, g, Ms, Hani):
    sqrt_term = np.sqrt((H + Hani) * (H + Hani + Ms))
    result = ((g * mue_B / h) * sqrt_term)
    return result

def kittel_equationALL(H_fields, g, Ms, *Hani_values):
    global LEN
    result = []
    Hani_values = np.array(Hani_values)
    H_fields = np.array(H_fields)
    for i, Hani in enumerate(Hani_values):
        H_subset = H_fields[i * LEN: (i + 1) * LEN]
        for H in H_subset:
            sqrt_term = np.sqrt((H + Hani) * (H + Hani + Ms))
            result.append((g * mue_B / h) * sqrt_term)
    return result

# Define linewidth equation
def linewidth_equation(f, alpha, dHinhomo):
    global Gfit
    result = ((alpha * 2 * np.pi * 1.054571817e-34)/(Gfit * 9.2740100783e-24) * f + dHinhomo)
    return result

def linewidth_equationALL(f_fields, alpha, *dHinhomo_values):
    global Gfit
    result = []
    dHinhomo_values = np.array(dHinhomo_values)
    f_fields = np.array(f_fields)
    for i, dHinhomo in enumerate(dHinhomo_values):
        f_subset = f_fields[i * LEN: (i + 1) * LEN]
        for f in f_subset:
            result.append((alpha * 4 * np.pi * 1.054571817e-34)/(Gfit * 9.2740100783e-24) * f + dHinhomo)
    return result  
        
    
def exportData(Hani_fit_dict):
    angles = list(Hani_fit_dict.keys())
    Hani_fit = list(Hani_fit_dict.values())
    new_data = np.column_stack((angles, Hani_fit))
    header = np.array(["Angle in °", "Hani_fit in T"])
    output_folder = '/home/julian/BA/dataForPython/FMR#3/H_resFields'
    output_filepath = os.path.join(output_folder, f'HaniFits.txt')
    np.savetxt(output_filepath, new_data, header='\t'.join(header), comments='', delimiter='\t', fmt='%.6e')




if __name__ == '__main__':
    main()