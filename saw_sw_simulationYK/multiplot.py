import os
from my_modules import *
from saw_simul_functions import *
import multiprocessing
from tqdm import tqdm
import copy
import re


input_folder = '/home/julian/BA/dataForPython'
output_folder = '/home/julian/BA/ColorMapTests'


def process_combination(params):
    combination, mySimulator = params
    copySimulator = copy.deepcopy(mySimulator)
    eps = combination
    copySimulator.calcH_dr(eps)
    copySimulator.calcP_abs()
    P_abs = copySimulator.P_abs
    Fields = copySimulator.Fields
    Angles = copySimulator.Angles
    non_zero_eps = {key: value for key, value in eps.items() if value != 0}
    name = f'Ray+{non_zero_eps}'

    cmPlot(P_abs, Fields, Angles, show=False, savePNG=True, name=name, outputfolder=output_folder)



def main():
    epsilonList = loadData()
    
    # Your parameter initialization here...
    Angles, Fields, params = GetParams()
    params_ = params[:12]

    mySimulator = SWcalculator(Fields, Angles, params_)
    mySimulator.calcPhi0()
    mySimulator.calcChi()


    # Create all combinations of the eps dictionary
    all_combinations = epsilonList
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



def loadData():
    filename = 'non_zero_eps_list.txt'
    filepath = os.path.join(input_folder, filename)
    with open(filepath, 'r') as file:
        data = file.readlines()

    # Define the default epsilon values
    default_eps = {'xx': 0, 'yy': 0, 'zz': 0, 'xy': 0, 'xz': 0, 'yz': 0}

    # Extract epsilon values from each line and create a list
    epsilon_list = []
    for line in data:
        match = re.search(r"Found Ray\+({.+})", line)
        if match:
            ray_str = match.group(1)
            ray_dict = eval(ray_str, {"__builtins__": None}, {})
            epsilon_values = {key: ray_dict.get(key, default_eps[key]) for key in default_eps}
            epsilon_list.append(epsilon_values)


    return epsilon_list


if __name__ == "__main__":
    main()