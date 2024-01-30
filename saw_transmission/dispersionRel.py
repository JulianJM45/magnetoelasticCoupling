import os
from my_modules import *
from scipy.optimize import curve_fit


# input_folder = r'C:\Users\Julian\Documents\MATLAB\Updated analysis sample#3\MatlabScript\960nm thickness estimated from quartz'
input_folder = '../dataForPython/DispersionRelation#3'

def main():
    kR, fR, vgR = GetData(name='Rayleigh')
    kS, fS, vgS = GetData(name='Sezawa')

    
    # plotKF(kR, fR, kS, fS, save=True)
    plotVG(fR, vgR, fS, vgS, save=False)


def plotVG(fR, vgR, fS, vgS, save=False):
    graph = Graph(width_cm=8.2)
    
    f_array = [0.9, 1.25, 2.43, 3.53, 4.49, 5.01]
    vg_array = [2765.0355721826363, 2469.654124939802, 2458.8934487701845, 2807.4903843454335, 2636.7483619200802, 2790.7314227986017]
    vg_array_err = [356.42200111722786, 267.02694844694895, 268.7286502879945, 293.4027838591837, 265.72190970375885, 313.8971728112079]
    vg_array = [x * 1e-3 for x in vg_array]
    vg_array_err = [x * 1e-3 for x in vg_array_err]

    colors = get_plot_colors(len(f_array))
    
    

    graph.add_plot(fR*1e-9, vgR*1e-3, label='Rayleigh', color='blue')
    graph.add_plot(fS*1e-9, vgS*1e-3, label='Sezawa', color='red')

    graph.add_scatter(f_array, vg_array, label='Messpunkte', marker='s', color=colors, s=20)
    for i in range(len(f_array)):
        graph.add_errorbar([f_array[i]], [vg_array[i]], yerror=[vg_array_err[i]], color=colors[i])
    # graph.add_errorbar(f_array, vg_array, yerror=vg_array_err, color=colors)

    graph.plot_Graph(xlabel='$f$\u2009(GHz)', ylabel='$v_G$\u2009(km/s)', name='v_gPlot', legend=False, save=save)


def plotKF(kR, fR, kS, fS, save=False):
    graph = Graph(width_cm=8.2)

    graph.add_plot(kR, fR*1e-9, label='Rayleigh', color='blue')
    graph.add_plot(kS, fS*1e-9, label='Sezawa', color='red')

    n=1
    k, f = get_kandf(fR, kR, n)
    print(f'for Rayleigh mode {n}: k = {k}, f = {f}')
    k, f = get_kandf(fS, kS, n)
    print(f'for Sezawa mode {n}: k = {k}, f = {f}')
    graph.add_vline(k*1e-6, label=f'n={n}', y=0.2, linestyle='-', color='green')
    
    n=3
    k, f = get_kandf(fR, kR, n)
    print(f'for Rayleigh mode {n}: k = {k}, f = {f}')
    k, f = get_kandf(fS, kS, n)
    print(f'for Sezawa mode {n}: k = {k}, f = {f}')
    graph.add_vline(k*1e-6, label=f'n={n}', y=0.2, linestyle='--', color='green')

    n=5
    k, f = get_kandf(fR, kR, n)
    print(f'for Rayleigh mode {n}: k = {k}, f = {f}')
    k, f = get_kandf(fS, kS, n)
    print(f'for Sezawa mode {n}: k = {k}, f = {f}')
    graph.add_vline(k*1e-6, label=f'n={n}', y=0.2, linestyle='-.', color='green')

    f_array = [0.9, 1.25, 2.43, 3.53, 4.49, 5.01]
    k_array = [k_n(1), k_n(1), k_n(3), k_n(3), k_n(5), k_n(5)]
    k_array = [x * 1e-6 for x in k_array]
    colors = get_plot_colors(len(f_array))
    graph.add_scatter(k_array, f_array, label='Messpunkte', marker='s', color=colors, s=20, zorder=2)

    

    graph.plot_Graph(xlabel='$k$\u2009(rad/μm)', ylabel='$f$\u2009(GHz)', name='Disprel', legend=False, save=save)



def GetData(name):
    filename = f'DispersionRelationFit{name}960nmZnO103nmYIG200nmDeadlayerIDT_FLLLS_c-axisangle0degrees.txt'
    input_filepath = os.path.join(input_folder, filename)
    data = np.loadtxt(input_filepath, dtype=float, delimiter=',')

    k = data[:, 0]*1e3 #1/m
    f = data[:, 1]*1e9 #Hz
    vg = data[:, 2]*1e3 #m/s

    return k, f, vg













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