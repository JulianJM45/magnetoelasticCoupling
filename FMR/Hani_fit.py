import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from my_modules import *
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

importfolder = r'C:\Users\Julian\Documents\BA\FMR#3'


def main():
    filename = 'HaniFits.txt'
    Angles, Hani = GetData(filename)

    fit1Hani = Fit1(Angles, Hani)

    Hani2 = Hani - fit1Hani
    fit2Hani = Fit2(Angles, Hani2)


    Hani3 = Hani2 -fit2Hani
    fit3Hani = Fit3(Angles, Hani3)

    Hani4 = Hani3 - fit3Hani
    fit4Hani = Fit4(Angles, Hani)

    Hani5 = Hani4 - fit4Hani

    
    Checkplot(Angles, Hani4, fit4Hani)
    GraphPlot(Angles, Hani5)


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

def Fit1Lin(Angles, Hani):
    initial_guess = [-0.0012, 1, 30, 0.0006, 1e-6]
    params, covariance = curve_fit(CosinusAndLinearFit, Angles, Hani, p0=initial_guess)
    a, b, c, d, m = params
    print(f'a: {a}')
    print(f'b: {b}')
    print(f'c: {c}')
    print(f'd: {d}')
    print(f'm: {m}')

    fit1Hani = CosinusAndLinearFit(Angles, a, b, c, d, m)

    return fit1Hani

# def CosinusFit(angle, a, b, c):
#     return (a*np.cos(b*(np.deg2rad(angle) - np.deg2rad(c))))

def SinusFit(angle, a, b, c):
    return (a*np.sin(b*(np.deg2rad(angle) - np.deg2rad(c))))

def BetragFit(angle, a, b):
    return (a * np.abs(angle) + b)

def CosinusFit(angle, a, b, c, d):
    return (a*(np.cos(b*(np.deg2rad(angle) - np.deg2rad(c))))**2+d)

def CosinusAndLinearFit(angle, a, b, c, d, m):
    return (a*(np.cos(b*(np.deg2rad(angle) - np.deg2rad(c))))**2+d + m*angle)

def SixfoldFit(angle, k1, k2):
    term1 = k1/12 *(7 - 8 + 4)
    term2 = k2/108 * (-24 + 45 - 24 + 4 + np.cos(6 * np.deg2rad(angle)))
    return (term1 + term2)

def GetData(filename):
    input_filepath = os.path.join(importfolder, filename)
    data = np.loadtxt(input_filepath, dtype=float, skiprows=1)
    x = data[:, 0]
    y = data[:, 1]
    return x, y



def Checkplot(Angles, Hani, Fit):
    sorted_data1 = sorted(zip(Angles, Hani), key=lambda pair: pair[0])
    sorted_data2 = sorted(zip(Angles, Fit), key=lambda pair: pair[0])
    Angles, Hani = zip(*sorted_data1)
    Angles, Fit = zip(*sorted_data2)
    plt.plot(Angles, Fit)
    plt.scatter(Angles, Hani)
    plt.show()
    plt.clf()











if __name__ == "__main__":
    main()