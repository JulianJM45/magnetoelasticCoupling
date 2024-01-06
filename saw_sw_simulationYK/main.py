import os
from my_modules import *
from saw_simul_functions import *
import time

name='Rayleigh'
Angles, Fields, params = GetParams(name)
# Angles = 20
# Fields = 20
params_ = params[:12]
eps = params[12]
# print(eps)
b1 = 4.579621001842745
b2 = 8.3127177727041
eps = {
            'xx': 0.2992345963780072-0.7587397581218336j,
            'yy': 0,
            'zz': 0,
            'xy': -0.059938004933110516-0.06072882210953735j,
            'xz': 0.6852275528945527+0.5125709160186992j,
            'yz': -0.16049272258898087+0.297642688093681j
        }




mySimulator = SWcalculator(Fields, Angles, params_)
mySimulator.calcPhi0()
mySimulator.calcChi()


start_time = time.time()

mySimulator.calcH_dr(b1, b2, eps)
mySimulator.calcP_abs(scale=True)
P_abs = mySimulator.P_abs

# P_abs = calculate(Fields, Angles, params)

elapsed_time = time.time() - start_time
print (f'elapsed time:  {elapsed_time} s')


cmPlot(P_abs, Fields, Angles, name=f'{name}_Fit1', save=True)




# print(CheckMax(Fields, Angles, P_abs))

# print(CheckSymmetrie(Fields, Angles, P_abs))


# params_ = params[:12]
# CombinationSweep(Angles, Fields, params_)




