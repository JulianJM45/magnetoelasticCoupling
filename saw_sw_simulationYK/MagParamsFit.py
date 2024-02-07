import numpy as np
import os 
from my_modules import *
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
# from h_res import allHres

input_folder = '../dataForPython/Field_Angle_Sweep#3'

name = 'Rayleigh'
name = 'Sezawa'

def main():
    Fields, Angles = loadData()

    mygraph = Graph()

    mygraph.add_scatter(Fields*1e3, Angles, label='Resonanzfelder', s=6)

    phiu_fit, mu_0Hani_fit, t_fit, A_fit, mu_0Ms_fit, g_fit = ResFit(Fields, Angles)
    Fields_fit = h_res.allHres(Angles, phiu=phiu_fit, mu_0Hani=mu_0Hani_fit, t=t_fit, A=A_fit, mu_0Ms=mu_0Ms_fit, g=g_fit)
    # Fields_fit = h_res.allHres(Angles, phiu=-0.944, mu_0Hani=0.0022591148169538253 , t=8.9491e-08, A=3.3e-12, mu_0Ms = 0.1721)
    # Fields_fit = h_res.allHres(Angles)
    r2_resonance = r2_score(Fields, Fields_fit)
    print(f'r2 score: {r2_resonance}')


    # Sort Fields_fit in ascending order of Angles
    sorted_indices = np.argsort(Angles)
    Fields_fit = Fields_fit[sorted_indices]

    mygraph.add_plot(Fields_fit*1e3, Angles[sorted_indices], label='Fit')


    mygraph.plot_Graph(save=False, show=True, legend=True, name=f'ResFields_{name}', xlabel='$|\mu_0H_0|$\u2009(mT)', ylabel='$\phi_H$\u2009(°)')


def loadData():
    # Define file paths
    if name == 'Rayleigh': filename = 'ResFields_Rayleigh.txt'
    elif name == 'Sezawa': filename = 'ResFields_Sezawa.txt'
    else: print('no file for this name')
    filepath = os.path.join(input_folder, filename)

    # Read the data from the .txt file
    data = np.loadtxt(filepath, dtype=float, skiprows=1)
    
    Angles = data[:, 0]
    Fields = data[:, 1]*0.001

    return Fields, Angles


def ResFit(Fields, Angles):
    initial_guess = getInitials()
 
    bounds = getBounds(initial_guess)

    # Perform the curve fit with bounds
    params, covariance = curve_fit(h_res.allHres, Angles, Fields, p0=initial_guess, bounds=bounds, maxfev=50000)
    # params, covariance = curve_fit(allHres, Angles, Fields, p0=initial_guess, maxfev=50000)

    phiu_fit, mu_0Hani_fit, t_fit, A_fit, mu_0Ms_fit, g_fit = params
    phiu_err, mu_0Hani_err, t_err, A_err, mu_0Ms_err, g_err = np.sqrt(np.diag(covariance)) # Get the standard deviations of the parameters (square roots of the diagonal of the covariance)

    print(f'phiu = {phiu_fit} +- {phiu_err}')
    print(f'mue0Hani = {mu_0Hani_fit} +- {mu_0Hani_err}')
    print(f't = {t_fit} +- {t_err}')
    print(f'A = {A_fit} +- {A_err}')
    print(f'mue0Ms = {mu_0Ms_fit} +- {mu_0Ms_err}')
    print(f'g = {g_fit} +- {g_err}')

    

    #Print the fitted parameters
    # print(f'Fitted g: {g_fit}')
    # print(f'Fitted mue0Ms: {mu_0Ms_fit}')
    # print(f'Fitted t: {t_fit}')
    # print(f'Fitted A: {A_fit}')
    # print(f'Fitted mue0Hani: {mu_0Hani_fit}')
    # print(f'Fitted phiu: {phiu_fit}')

    # return phiu_fit, mu_0Hani_fit, t_fit, A_fit
    return phiu_fit, mu_0Hani_fit, t_fit, A_fit, mu_0Ms_fit, g_fit


def getInitials():
    # initial guesses

    t = 1 * 100e-9 #20%
    A = 3.7e-12 #+- 0.4e-12
    mu_0Hani = 0.0015 #+- 5mT
    phiu = 38.5 #+- 180°

    mu_0Ms=0.18381376682738004
    g = 2.026899502405165
    
    # initial_guess = [phiu, mu_0Hani, t, A]
    initial_guess = [phiu, mu_0Hani, t, A, mu_0Ms, g]

    return initial_guess


def getBounds(fitparams):
    # Define bounds as a percentage of the initial values
    bounds_factor = 0.10  # 10 percent
    lower_bounds = [param - np.abs(param) * bounds_factor for param in fitparams]
    upper_bounds = [param + np.abs(param) * bounds_factor for param in fitparams]

    # #set bounds for phiu
    lower_bounds[0] = fitparams[0] - 180
    upper_bounds[0] = fitparams[0] + 180

    #set bounds for mue0Hani
    lower_bounds[1] = 0
    upper_bounds[1] = 25e-3

    #set bounds for t
    lower_bounds[2] = 0.8 * fitparams[2]  # 20%   
    upper_bounds[2] = 1.2 * fitparams[2]  # 20%

    # set bounds for A
    lower_bounds[3] = fitparams[3] - 0.4e-12 
    upper_bounds[3] = fitparams[3] + 0.4e-12

    #set bounds for mu_0Ms
    lower_bounds[4] = 0
    upper_bounds[4] = 2

    #set bounds for g
    lower_bounds[5] = 2.0
    upper_bounds[5] = 2.1
    
    bounds = (lower_bounds, upper_bounds)

    return bounds



if __name__=='__main__':
    main()