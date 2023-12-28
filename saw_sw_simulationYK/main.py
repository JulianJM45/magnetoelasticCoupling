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
# mySimulator = SWcalculator(Fields, Angles, params_)
# mySimulator.calcPhi0()
# mySimulator.calcChi()
# mySimulator.calcH_dr(eps)
# mySimulator.calcP_abs()



# start_time = time.time()

P_abs = calculate(Fields, Angles, params)
# P_abs = mySimulator.P_abs

# elapsed_time = time.time() - start_time
# print (f'elapsed time:  {elapsed_time} s')


cmPlot(P_abs, Fields, Angles, name=name)




# print(CheckMax(Fields, Angles, P_abs))

# print(CheckSymmetrie(Fields, Angles, P_abs))


# params_ = params[:12]
# CombinationSweep(Angles, Fields, params_)




