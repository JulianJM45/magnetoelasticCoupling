import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from my_modules import *
from scipy.optimize import curve_fit

#Define file path
# input_folder= r'C:\Users\Julian\Documents\MATLAB\Updated analysis sample#3\MatlabScriptProcessedOutputs'
input_folder = '/home/julian/BA/dataForPython/DispersionRelation#3'
# output_filepath = r'C:\Users\Julian\Pictures\DispersionRel.png'
outputfolder = '/home/julian/BA/pictures'


def main():
    kR, fR = GetData(name='Rayleigh')
    kS, fS = GetData(name='Sezawa')

    graph = Graph()

    graph.add_plot(kR*1e-6, fR*1e-9, label='Rayleigh')
    graph.add_plot(kS*1e-6, fS*1e-9, label='Sezawa', color='red')

    n=1
    k, f = get_kandf(fR, kR, n)
    print(f'for Rayleigh mode {n}: k = {k}, f = {f}')
    k, f = get_kandf(fS, kS, n)
    print(f'for Sezawa mode {n}: k = {k}, f = {f}')
    graph.add_vline(k*1e-6, label=f'n={n}', linestyle='-', color='green')
    
    n=3
    k, f = get_kandf(fR, kR, n)
    print(f'for Rayleigh mode {n}: k = {k}, f = {f}')
    k, f = get_kandf(fS, kS, n)
    print(f'for Sezawa mode {n}: k = {k}, f = {f}')
    graph.add_vline(k*1e-6, label=f'n={n}', linestyle='--', color='green')

    n=5
    k, f = get_kandf(fR, kR, n)
    print(f'for Rayleigh mode {n}: k = {k}, f = {f}')
    k, f = get_kandf(fS, kS, n)
    print(f'for Sezawa mode {n}: k = {k}, f = {f}')
    graph.add_vline(k*1e-6, label=f'n={n}', linestyle='-.', color='green')



    graph.plot_Graph(xlabel='$k$ in $10^{6} \\frac{1}{m}$', ylabel='$f$ in GHz', legend=True, outputfolder=outputfolder, safe=True)





def GetData(name):
    if name == 'Rayleigh': filename = 'DispersionRelationRayleigh960nm.txt'
    elif name == 'Sezawa': filename = 'DispersionRelationSezawa960nm.txt'
    else: print('false name')
    input_filepath = os.path.join(input_folder, filename)
    data = np.loadtxt(input_filepath, dtype=float, delimiter=',')

    # frequency = data[:, 0]*1e9 #Hz
    # v_ph = data[:, 1]*1e3   #m/s
    k = data[:, 0]*1e6
    f = data[:, 1]*1e9 #Hz
    return k, f


def get_kandf(farray, karray, n):
    a, b = fit(farray, karray)
    k = k_n(n)
    f = quadratic(k, a, b)

    return k, f



def fit(f, k):
    params, covar = curve_fit(quadratic, k, f)
    a, b = params
    return a, b


# Define the function to fit the data
def quadratic(x, a, b):
    return a*x + b


def k_n(n):
    # IDT fingerabstand
    b = 700 * 1e-9 #m
    k = n * np.pi/(2*b)

    return k


if __name__ == '__main__':
    main()




'''



print(f'k for order {n}: {k}')







# Fit the data using curve_fit
popt, pcov = curve_fit(func, fR, kR)
popt, pcov = curve_fit(func, fS, kS)

# Calculate the value of kR for fR = 2.43
f0 = 2.43
k0 = func(f0, *popt)
CsawR = 2* np.pi * f0/k0*1e3

f1 = 3.53
k1 = func(f1, *popt)
CsawS = 2* np.pi * f1/k1*1e3

# Print the result
print('kR for fR = {}: {} '.format(f0, k0))
print('CsawR for fR = {}: {} in m/s'.format(f0, CsawR))

print('kS for fS = {}: {} '.format(f1, k1))
print('CsawS for fS = {}: {} in m/s'.format(f1, CsawS))





'''