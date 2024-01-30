import os
from my_modules import *
import numpy as np
from scipy.optimize import curve_fit
import platform

if platform.system() == 'Linux': input_folder = '/home/julian/Seafile/BAJulian/dataForPython/FMR#3/H_resFields'
elif platform.system() == 'Windows': input_folder = r'C:\Users\Julian\Seafile\BAJulian\dataForPython\FMR#3\H_resFields'

alpha = 0

def main():
    filename = 'HaniFits.txt'
    Angles, Hani = GetData(filename)
    Angles = np.array(Angles)-alpha

    # fit1Hani = CosinusAndLinearFit(Angles, Hani)
    # fit1Hani = CosinusFit(Angles, Hani)
    fit1Hani = doubleCosinusFit(Angles, Hani)



    Hani2 = Hani - fit1Hani
    # fit2Hani = SixfoldAndBetragFit(Angles, Hani2)
    # fit2Hani = SixfoldFit(Angles, Hani2)


    
    Checkplot(Angles, Hani, fit1Hani, save=True, name='Hani_fit')
    # GraphPlot(Angles, Hani2, save=False, name='Hani_sixfold', xlabel='Angle in °', ylabel='$\~{H}_\mathrm{ani}$ in mT')


def Checkplot(Angles, Hani, Fit, save=False, name='Hani_fit'):
    mygraph = Graph()
    mygraph.go_polar()
    mygraph.add_plot(np.radians(Angles), Fit*1e3, color='r', label='Fitkurve für $H_\mathrm{ani}$')
    mygraph.add_scatter(np.radians(Angles), Hani*1e3, label='gefittete Datenpunkte für $H_\mathrm{ani}$')

    mygraph.plot_Graph(save=save, legend=False, name=name, xlabel='$\phi_H$', ylabel='$\mu_0H_\mathrm{ani}$\u2009(mT)')


def doubleCosinusFit(Angles, Hani):
    initial_guess = [1, 135, 1, 30, 0]
    params, covariance = curve_fit(doubleCosinusCalc, Angles, Hani, p0=initial_guess)
    a1, c1, a2, c2, d = params
    # Calculate the standard errors
    standard_errors = np.sqrt(np.diag(covariance))

    # Print the parameters and their standard errors
    for i, param in enumerate(['a1', 'c1', 'a2', 'c2', 'd']):
        print(f'{param}: {params[i]} ± {standard_errors[i]}')

    fit1Hani = doubleCosinusCalc(Angles, a1, c1, a2, c2, d)

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
    # alpha = int(c)

    fit1Hani = CosinusAndLinearCalc(Angles, a, b, c, d, m)

    return fit1Hani

def CosinusFit(Angles, Hani):
    initial_guess = [-0.0012, 1, 30, 0.0006]
    params, covariance = curve_fit(CosinusCalc, Angles, Hani, p0=initial_guess)
    a, b, c, d = params
    print(f'a: {a}')
    print(f'b: {b}')
    print(f'c: {c}')
    print(f'd: {d}')

    fit1Hani = CosinusCalc(Angles, a, b, c, d)

    return fit1Hani

def SixfoldFit(Angles, Hani):
    initial_guess = [1, 1, 0]
    params, covariance = curve_fit(SixfoldCalc, Angles, Hani, p0=initial_guess)
    k1, k2, phi0 = params
    print(f'k1: {k1}')
    print(f'k2: {k2}')
    print(f'phi0: {phi0}')

    fit1Hani = SixfoldCalc(Angles, k1, k2, phi0)

    return fit1Hani

def SixfoldAndBetragFit(Angles, Hani2):
    initial_guess = [1, 1, 0]
    # params, covariance = curve_fit(SixfoldCalc, Angles, Hani2, p0=initial_guess)
    params, covariance = curve_fit(SixfoldAndBetragCalc, Angles, Hani2)
    k1, k2, phi0, a = params
    print(f'k1: {k1}')
    print(f'k2: {k2}')
    print(f'phi0: {phi0}')
    print(f'a: {a}')

    fit2Hani = SixfoldAndBetragCalc(Angles, k1, k2, phi0, a)

    return fit2Hani





def doubleCosinusCalc(angle, a1, c1, k2, c2, k1):
    b1 = 2
    b2 = 6
    return (a1*np.cos(b1*(np.deg2rad(angle - c1)))+ k2 *np.cos(b2*(np.deg2rad(angle - c2)))+k1)


def SinusCalc(angle, a, b, c):
    return (a*np.sin(b*(np.deg2rad(angle) - np.deg2rad(c))))

def BetragCalc(angle, a, b):
    return (a * np.abs(angle) + b)

def CosinusCalc(angle, a, b, c, d):
    return (a*(np.cos(b*(np.deg2rad(angle) - np.deg2rad(c))))**2+d)

def CosinusAndLinearCalc(angle, a, b, c, d, m):
    return (a*(np.cos(b*(np.deg2rad(angle) - np.deg2rad(c))))**2+d + m*angle)

def SixfoldCalc(angle, k1, k2, phi0=0):
    # term1 = k1/12 *(7 - 8 + 4)
    # term2 = k2/108 * (-24 + 45 - 24 + 4 + np.cos(6 * np.deg2rad(angle)))
    term = k1 + k2 * np.cos(6 * np.deg2rad(angle-phi0))
    # return (term1 + term2)
    return term

def SixfoldAndBetragCalc(angle, k1, k2, phi0=0, a=1):
    term1 = k1 + k2 * np.cos(6 * np.deg2rad(angle-phi0))
    term2 = a * np.abs(angle-phi0) + k1
    return (term1 + term2)

def GetData(filename):
    input_filepath = os.path.join(input_folder, filename)
    data = np.loadtxt(input_filepath, dtype=float, skiprows=1)
    x = data[:, 0]
    y = data[:, 1]
    return x, y










if __name__ == "__main__":
    main()