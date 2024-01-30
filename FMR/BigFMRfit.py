import pandas as pd
import os
import numpy as np
from my_modules import *
import scipy
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
# import platform

# if platform.system() == 'Linux': input_folder = '/home/julian/BA/dataForPython/FMR#3/ResonanceData'
# elif platform.system() == 'Windows': input_folder = r'C:\Users\Julian\Seafile\BAJulian\dataForPython\FMR#3\ResonanceData'

input_folder = '../dataForPython/FMR#3/ResonanceData'

LEN = 0

Gfit = 2.026899502405165

#constants
h = scipy.constants.h
mue_B = scipy.constants.e * scipy.constants.hbar /  (2* scipy.constants.m_e)


def main():
    frequency, H, H_err, dHres, dHres_err = getData()

    # g_fit, Ms_fit, Hani_fit_dict, g_error, Ms_error, Hani_error_dict = KittelFit(frequency, H)

    # # meanHani = np.mean(list(Hani_fit_dict.values()))
    # # f = 3.53e9
    # # print(invert_kittel_equation(f, g_fit, Ms_fit, meanHani))
    # # exportData(Hani_fit_dict)

    # KittelPlot(frequency, H, H_err, g_fit, Ms_fit, Hani_fit_dict, angle=180, save=True)

    alpha_fit, dHinhomo_fit_dict, alpha_error, dHinhomo_error_dict = LinewidthFit(frequency, dHres)
    LinewidthPlot(frequency, dHres, dHres_err, alpha_fit, dHinhomo_fit_dict, angle=180, save=True)

    # dHinhomoPlot(dHinhomo_fit_dict, save=True)

    





def LinewidthFit(frequency, dHres):
    angle_count = len(frequency)
    all_f_array = np.concatenate(list(frequency.values()))
    all_dHres_array = np.concatenate(list(dHres.values()))

    # print(frequency.keys())
    #initial guesses for fit:
    alpha1 = 0.00078
    dHinhomo1 = 0.0035
    initial_guess_linewidth = np.hstack([alpha1, np.full(angle_count, dHinhomo1)])

    params, covariance = curve_fit(linewidth_equationALL, all_f_array, all_dHres_array, p0=initial_guess_linewidth)
    alpha_fit, *dHinhomo_fit_array = params
    dHinhomo_fit_dict = dict(zip(frequency.keys(), dHinhomo_fit_array))
    # print(dHinhomo_fit_dict.keys())

    # Calculate the errors
    errors = np.sqrt(np.diag(covariance))
    alpha_error, *dHinhomo_error_array = errors
    dHinhomo_error_dict = dict(zip(frequency.keys(), dHinhomo_error_array))


    print(f"Fitted alpha: {alpha_fit} ± {alpha_error}")
    for angle, dHinhomo_fit in dHinhomo_fit_dict.items():
        dHinhomo_error = dHinhomo_error_dict[angle]
        # print(f"Fitted dHinhomo for angle {angle}: {dHinhomo_fit} ± {dHinhomo_error}")

    return alpha_fit, dHinhomo_fit_dict, alpha_error, dHinhomo_error_dict

def LinewidthPlot(frequency, dHres, dHres_err, alpha_fit, dHinhomo_fit_dict, angle=0, show=True, save=False):
    mygraph = Graph(width_cm=8.2)
    colors = get_plot_colors(len(dHinhomo_fit_dict))

    # for i, (angle, dHinhomo_fit) in enumerate(dHinhomo_fit_dict.items()):
    #     color = colors[i]
        
    
    f = frequency[angle]
    dH_res = dHres[angle]
    dH_res_err = dHres_err[angle]   

    
    mygraph.add_errorbar(f*1e-9, dH_res*1e3, xerror=None, yerror=dH_res_err*1e3, label='Measured Data', s=2)

    dHinhomo_fit = dHinhomo_fit_dict[angle]
    dH_res_fit = linewidth_equation(f, alpha_fit, dHinhomo_fit)

    mygraph.add_plot(f*1e-9, dH_res_fit*1e3, label=f'Fitted Data for {angle}°')

    
    # dH_res_fit = linewidth_equation(f, alpha_fit, dHinhomo_fit_dict[angle])
    # mygraph.add_plot(f*1e-9, dH_res_fit*1e3, label='Fitted Data')

    # mygraph.ColorMap=True

    mygraph.plot_Graph(save=save, show=show, legend=False, xlabel='$f$\u2009(GHz)', ylabel='$\mu_0\Delta H_\mathrm{res}$\u2009(mT)', name=f'dHresPlot{angle}', title='b)')



def dHinhomoPlot(dHinhomo_fit_dict, save=False):
    graph=Graph()
    graph.go_polar()
    angles = list(dHinhomo_fit_dict.keys())
    dHin = np.array(list(dHinhomo_fit_dict.values()))
    graph.add_scatter(np.radians(angles), dHin*1e3, label='Fitted Data', s=2)
    graph.plot_Graph(save=save, xlabel='$\phi_H$', ylabel='$\Delta H_\mathrm{inhomo}$\u2009(mT)', legend=False, name='dHinhomoPlot')
    # GraphPlot(angles, dHin*1e3, save=save, name="deltaHinhomo", xlabel='Angles in °', ylabel='$\mu_0 \Delta H_\mathrm{inhomo}$ in mT')

    

def KittelPlot(frequency, H, H_err, g_fit, Ms_fit, Hani_fit_dict, angle=0, save=False):
    Hani = Hani_fit_dict[angle]
    f = frequency[angle]
    H_array = H[angle]
    H_err = H_err[angle]

    f_fit = kittel_equation(H_array, g_fit, Ms_fit, Hani)


    mygraph = Graph(width_cm=8.2)
    mygraph.add_plot(H_array, f_fit*1e-9, label='Fitted Data')
    mygraph.add_errorbar(H_array, f*1e-9, xerror=H_err, yerror=None, label='Measured Data', s=2)

    mygraph.plot_Graph(save=save, legend=None, xlabel='$\mu_0 H_0$\u2009(T)', ylabel='$f$\u2009(GHz)', name=f'KittelPlot{angle}', title='a)')

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
    params, cov = curve_fit(kittel_equationALL, all_H_array, all_f_array, initial_guess_kittel)
    g_fit, Ms_fit, *Hani_fit_array = params
    Hani_fit_dict = dict(zip(frequency.keys(), Hani_fit_array))

    # Calculate the errors for the fit parameters
    errors = np.sqrt(np.diag(cov))
    g_error, Ms_error, *Hani_error_array = errors
    Hani_error_dict = dict(zip(frequency.keys(), Hani_error_array))

    global Gfit
    Gfit = g_fit

    # Print the fitted parameters and their errors
    print(f"Fitted g: {g_fit} ± {g_error}")
    print(f"Fitted Ms: {Ms_fit} ± {Ms_error}")
    for angle, Hani_fit in Hani_fit_dict.items():
        Hani_error = Hani_error_dict[angle]
        # print(f"Fitted Hani for angle {angle}: {Hani_fit} ± {Hani_error}")
    
    return g_fit, Ms_fit, Hani_fit_dict, g_error, Ms_error, Hani_error_dict





def getData():
    frequency = {}
    H = {}
    H_err = {}
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
        H_err[angle] = data['Hres err1']
        dHres[angle]  = data['dH1']
        dHres_err[angle] = data['dH err1']

     # Sort the angles in ascending order
    angles.sort()

    # Create sorted dictionaries using the sorted angles
    sorted_frequency = {angle: frequency[angle] for angle in angles}
    sorted_H = {angle: H[angle] for angle in angles}
    sorted_H_err = {angle: H_err[angle] for angle in angles}
    sorted_dHres = {angle: dHres[angle] for angle in angles}
    sorted_dHres_err = {angle: dHres_err[angle] for angle in angles}

    # Update the global variable LEN
    global LEN
    LEN = len(sorted_frequency[angles[0]])

    return sorted_frequency, sorted_H, sorted_dHres_err, sorted_dHres, sorted_dHres_err


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
    result = ((alpha * 4 * np.pi * 1.054571817e-34)/(Gfit * 9.2740100783e-24) * f + dHinhomo)
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