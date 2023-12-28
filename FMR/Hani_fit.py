import os
from my_modules import *
import numpy as np
from scipy.optimize import curve_fit

# importfolder = r'C:\Users\Julian\Documents\BA\FMR#3'
importfolder = '/home/julian/BA/dataForPython/FMR#3/H_resFields'
output_folder = '/home/julian/BA/pictures'

alpha = 0

def main():
    filename = 'HaniFits.txt'
    Angles, Hani = GetData(filename)

    fit1Hani = CosinusAndLinearFit(Angles, Hani)

    global alpha
    Angles = np.array(Angles)-alpha


    Hani2 = Hani - fit1Hani

    # fit2Hani = Fit2(Angles, Hani2)


    # Hani3 = Hani2 -fit2Hani
    # fit3Hani = Fit3(Angles, Hani3)

    # Hani4 = Hani3 - fit3Hani
    # fit4Hani = Fit4(Angles, Hani)

    # Hani5 = Hani4 - fit4Hani

    
    # Checkplot(Angles, Hani, fit1Hani)
    GraphPlot(Angles, Hani2, save=True, name='Hani_sixfold', xlabel='Angle in °', ylabel='$\~{H}_\mathrm{ani}$ in mT')


def Fit4(Angles, Hani4):
    initial_guess = [1, 4, 1]
    params, covariance = curve_fit(SinusFit, Angles, Hani4, p0=initial_guess)
    a, b, c = params

    fit4Hani = SinusFit(Angles, a, b, c)

    return fit4Hani

def Fit3(Angles, Hani3):
    initial_guess = [1, - 0.00015]
    params, covariance = curve_fit(BetragFit, Angles, Hani3, p0=initial_guess)
    a, b = params

    fit3Hani = BetragFit(Angles, a, b)

    return fit3Hani


def Fit2(Angles, Hani2):
    initial_guess = [1, 1]
    params, covariance = curve_fit(SixfoldFit, Angles, Hani2, p0=initial_guess)
    k1, k2 = params
    print(f'k1: {k1}')
    print(f'k2: {k2}')

    fit2Hani = SixfoldFit(Angles, k1, k2)

    return fit2Hani
    


def Fit1(Angles, Hani):
    # initial_guess = [-0.0006, 2, 30]
    initial_guess = [-0.0012, 1, 30, 0.0006]
    # initial_guess = [-0.0012, 1, 30, 0.0006, 1e-6]

    params, covariance = curve_fit(CosinusFit, Angles, Hani, p0=initial_guess)
    # params, covariance = curve_fit(CosinusAndLinearFit, Angles, Hani, p0=initial_guess)
    a, b, c, d = params

    # a, b, c, d, m = params
    print(f'a: {a}')
    print(f'b: {b}')
    print(f'c: {c}')
    print(f'd: {d}')
    # print(f'm: {m}')

    # fit1Hani = CosinusFit(Angles, a, b, c)
    fit1Hani = CosinusFit(Angles, a, b, c, d)
    # fit1Hani = CosinusAndLinearFit(Angles, a, b, c, d, m)

    return fit1Hani

def CosinusAndLinearFit(Angles, Hani):
    initial_guess = [-0.0012, 1, 30, 0.0006, 1e-6]
    params, covariance = curve_fit(CosinusAndLinearCalc, Angles, Hani, p0=initial_guess)
    a, b, c, d, m = params
    print(f'a: {a}')
    print(f'b: {b}')
    print(f'c: {c}')
    print(f'd: {d}')
    print(f'm: {m}')

    global alpha
    alpha = int(c)

    fit1Hani = CosinusAndLinearCalc(Angles, a, b, c, d, m)

    return fit1Hani

# def CosinusFit(angle, a, b, c):
#     return (a*np.cos(b*(np.deg2rad(angle) - np.deg2rad(c))))

def SinusCalc(angle, a, b, c):
    return (a*np.sin(b*(np.deg2rad(angle) - np.deg2rad(c))))

def BetragCalc(angle, a, b):
    return (a * np.abs(angle) + b)

def CosinusCalc(angle, a, b, c, d):
    return (a*(np.cos(b*(np.deg2rad(angle) - np.deg2rad(c))))**2+d)

def CosinusAndLinearCalc(angle, a, b, c, d, m):
    return (a*(np.cos(b*(np.deg2rad(angle) - np.deg2rad(c))))**2+d + m*angle)

def SixfoldCalc(angle, k1, k2):
    term1 = k1/12 *(7 - 8 + 4)
    term2 = k2/108 * (-24 + 45 - 24 + 4 + np.cos(6 * np.deg2rad(angle)))
    return (term1 + term2)

def GetData(filename):
    input_filepath = os.path.join(importfolder, filename)
    data = np.loadtxt(input_filepath, dtype=float, skiprows=1)
    x = data[:, 0]
    y = data[:, 1]
    return x, y



def Checkplot(Angles, Hani, Fit, save=False, name=''):
    mygraph = Graph()
    mygraph.add_plot(Angles, Fit*1e3, color='r', label='$cos²$ + linear Fit')
    mygraph.add_scatter(Angles, Hani*1e3, label='Messdaten')
    mygraph.plot_Graph(safe=True, legend=True, name='Hani_fit', xlabel='Angle in °', ylabel='$H_\mathrm{ani}$ in mT')











if __name__ == "__main__":
    main()