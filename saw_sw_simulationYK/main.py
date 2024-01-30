import os
from my_modules import *
from saw_simul_functions import *
import time

input_folder = '../dataForPython/Field_Angle_Sweep#3'

name='Rayleigh'
name= 'Sezawa'

def main():
    Angles, Fields, params = GetParams(name)
    # Angles = 20
    # Fields = 20
    params_ = params[:10]
    alpha = params[0]
    b1, b2, eps = params[10:13]

    # # best fit for Rayleigh
    alpha =  0.013
    # eps['xx'] =  (1-0.77j)
    # eps['xy'] =  (0.18+0.26j)
    # eps['xz'] =  (0.39+0.58j)
    # eps['yz'] =  (0.06-0.04j)

    # best fit for Sezawa
    alpha =  0.0064
    eps['xx'] =  (0.81+0.2j)
    eps['xy'] =  (0.09-0.36j)
    eps['xz'] =  (0.21-1j)
    eps['yz'] =  (0.15+0.05j)




    start_time = time.time()

    mySimulator = SWcalculator(Fields, Angles, params_)
    mySimulator.calcPhi0()
    mySimulator.calcChi(alpha=alpha)
    mySimulator.calcH_dr(b1, b2, eps)
    mySimulator.calcP_abs(scale=True)
    P_abs = mySimulator.P_abs-1

    # P_abs = calculate(Fields, Angles, params)

    elapsed_time = time.time() - start_time
    print (f'elapsed time:  {elapsed_time} s')


    cmPlot(P_abs, Fields, Angles, cmap='hot', name=f'{name}_Fit2', save=False)



    # print(CheckMax(Fields, Angles, P_abs))

    # print(CheckSymmetrie(Fields, Angles, P_abs))


    # params_ = params[:12]
    # CombinationSweep(Angles, Fields, params_)





    



if __name__ == "__main__":
    main()