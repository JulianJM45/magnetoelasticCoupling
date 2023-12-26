import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from my_modules import *
from saw_sw_simulationYK.saw_simul_functions import *
import itertools
import multiprocessing
from tqdm import tqdm
import copy

epsilonList = []


def process_combination(params):
    combination, mySimulator = params
    copySimulator = copy.deepcopy(mySimulator)
    # Angles = np.linspace(-90, 90, num=41)
    # Fields = np.linspace(-50, 50, num=51)
    # alpha, AniType, mue0Hani, phiu, A, g, mue0Ms, k, b1, b2, f, t = params_
    eps = combination
    # params = [alpha, AniType, mue0Hani, phiu, A, g, mue0Ms, k, eps['xx'], eps['yy'], eps['yy'], eps['xy'], eps['xz'], eps['yz'], b1, b2, f, t]
    copySimulator.calcH_dr(eps)
    copySimulator.calcP_abs()
    # P_abs = calculate(Fields, Angles, params)
    P_abs = copySimulator.P_abs
    Fields = copySimulator.Fields
    Angles = copySimulator.Angles
    non_zero_eps = {key: value for key, value in eps.items() if value != 0}
    name = f'Ray+{non_zero_eps}'

    if CheckMax(Fields, Angles, P_abs):
        if CheckSymmetrie(Fields, Angles, P_abs):
            epsilonList.append(non_zero_eps)
            print(f'Found {name}')
        else: print('skipped -- no symmetrie')

        


def main():
    # Your parameter initialization here...
    Angles, Fields, params = GetParams()
    params_ = params[:12]

    mySimulator = SWcalculator(Fields, Angles, params_)
    mySimulator.calcPhi0()
    mySimulator.calcChi()

    

    eps = {
    'xx': 1,
    'yy': 0,
    'zz': 0,
    'xy': 0,
    'xz': 0,
    'yz': 0
    }

    # Values for real and imaginary parts to sweep through
    values = [-1, 0.5, 0, 0.5, 1]

    # Generate all combinations for each key (excluding zz)
    combinations_per_key = {key: [complex(real, imag) for real in values for imag in values] if key not in ['yy', 'zz'] else [0] for key in eps}

    # Create all combinations of the eps dictionary
    all_combinations = [dict(zip(eps.keys(), combination)) for combination in itertools.product(*combinations_per_key.values())]
    num_combinations = len(all_combinations)

    # Create a pool of processes
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)


    # Prepare parameters for parallel processing
    # params_list = [(combination, params_) for combination in all_combinations]
    params_list = [(combination, mySimulator) for combination in all_combinations]

    # Use the pool to process combinations in parallel

    with tqdm(total=len(all_combinations), desc="Processing", position=0, leave=True) as pbar:
       for _ in pool.imap_unordered(process_combination, params_list):
           pbar.update(1)
    # Close the pool to free up resources
    pool.close()
    pool.join()

       # Save the result_list to a .txt file
    with open('non_zero_eps_list.txt', 'w') as file:
        for item in epsilonList:
            file.write(f"{item}\n")









if __name__ == "__main__":
    main()