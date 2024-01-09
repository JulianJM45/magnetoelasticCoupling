import numpy as np
import os 
from my_modules import *
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from h_res import allHres



# input_filepath = r'C:\Users\Julian\Documents\BA\Field_and_Angle_Sweep#3\M06\ResFields_Rayleigh.txt'
# input_folder = '/home/julian/BA/dataForPython/Field_Angle_Sweep#3'
input_folder = r'C:\Users\Julian\Documents\BA\Field_and_Angle_Sweep#3\M06'

name = 'Rayleigh'

def main():
    Fields, Angles = loadData()

    mygraph = Graph()

    mygraph.add_scatter(Fields*1e3, Angles, label='Resonanzfelder')

    # t_fit, g_fit, mue0Hani_fit, mue0Ms_fit = ResFit(Fields, Angles)
    # Fields_fit = allHres(Angles, t_fit, g_fit, mue0Hani_fit, mue0Ms_fit)
    # Atest = -6.19632152482511e-11
    t, g, mue0Hani, mue0Ms = getInitials()
    Fields_fit = allHres(Angles, t, g, mue0Hani, mue0Ms)
    mygraph.add_plot(Fields_fit*1e3, Angles, label='Fit')

    # r2_resonance = r2_score(Fields, Fields_fit)
    # print(f'r2 score: {r2_resonance}')

    

    mygraph.plot_Graph(save=True, legend=True, name=f'ResFields_{name}', xlabel='$\mu_0 H$ in mT', ylabel='Angle in °')




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
    params, covariance = curve_fit(allHres, Angles, Fields, p0=initial_guess, bounds=bounds, maxfev=50000)


    # Extract the fitted parameters
    t_fit, g_fit, mue0Hani_fit, mue0Ms_fit = params

    # Print the fitted parameters
    print(f'Fitted t: {t_fit}')
    # print(f'Fitted A: {A_fit}')
    print(f'Fitted g: {g_fit}')
    print(f'Fitted mue0Hani: {mue0Hani_fit}')
    print(f'Fitted mue0Ms: {mue0Ms_fit}')

    return t_fit, g_fit, mue0Hani_fit, mue0Ms_fit





def getInitials():
    # initial guesses
    t = 1 * 100e-9 *0.8#20%
    g = 2.027 #2-2.1
    mue0Hani = 0 #+- 5mT
    mue0Ms =  0.18381 # 10%

    initial_guess = [t, g, mue0Hani, mue0Ms]

    return initial_guess


def getBounds(fitparams):
    # Define bounds as a percentage of the initial values
    bounds_factor = 0.10  # 10 percent
    bounds_factor2 = 500
    lower_bounds = [param - np.abs(param) * bounds_factor for param in fitparams]
    upper_bounds = [param + np.abs(param) * bounds_factor for param in fitparams]
    #set high bounds for mue0Hani
    for i in [2, 3]:
        lower_bounds[i] = fitparams[i] - np.abs(fitparams[i]) * bounds_factor2
        upper_bounds[i] = fitparams[i] + np.abs(fitparams[i]) * bounds_factor2
    bounds = (lower_bounds, upper_bounds)

    return bounds



if __name__=='__main__':
    main()