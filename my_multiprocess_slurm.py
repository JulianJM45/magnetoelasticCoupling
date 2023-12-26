from saw_sw_simulationYK import calculate, GetParams
import numpy as np
import itertools
from mpi4py.futures import MPIPoolExecutor
from tqdm import tqdm


def MinMaxScaling(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    scaled_matrix = (matrix - min_val) / (max_val - min_val)
    return scaled_matrix


def CheckMax(X, Y, Z):
    min_indices = np.unravel_index(np.argmax(Z), Z.shape)
    if isinstance(X, np.matrix):
        # print('is Matrix')
        angle_value = int(Y[min_indices])
        field_value = int(X[min_indices])
        
    else:
        # print('is array')
        angle_value = int(np.sort(np.unique(Y))[min_indices[0]])
        field_value = int(np.sort(np.unique(X))[min_indices[1]])
        
    angle_interval = range(10, 60)
    field_interval = range(15, 40)
    # print(field_value, angle_value)
    
    is_within_intervals = (
        (angle_value in angle_interval and -field_value in field_interval) or
        (-angle_value in angle_interval and field_value in field_interval)
    )
    # print(is_within_intervals)
    return is_within_intervals


def CheckSymmetrie(Fields, Angles, P_abs):
    pos_field_indices = np.where(Fields > 0)
    neg_field_indices = np.where(Fields < 0)
    # pos_angle_indices = np.where(Angles > 0)
    # neg_angle_indices = np.where(Angles < 0)
    
    sum_of_positive_P_abs = np.sum(P_abs[:, pos_field_indices])    
    sum_of_negative_P_abs = np.sum(P_abs[:, neg_field_indices])
    difference1 = np.abs(sum_of_positive_P_abs - sum_of_negative_P_abs)
    # print(difference1)
    if difference1 < 20:
        return True
    else:
        
        return False
epsilonList = []


def process_combination(params):
    combination, params_ = params
    Angles = np.linspace(-90, 90, num=41)
    Fields = np.linspace(-50, 50, num=51)
    alpha, AniType, mue0Hani, phiu, A, g, mue0Ms, k, b1, b2, f, t = params_
    eps = combination
    non_zero_eps = {key: value for key, value in eps.items() if value != 0}
    name = f'Ray+{non_zero_eps}'
    params = [alpha, AniType, mue0Hani, phiu, A, g, mue0Ms, k, eps['xx'], eps['yy'], eps['yy'], eps['xy'], eps['xz'], eps['yz'], b1, b2, f, t]
    P_abs = calculate(Angles, Fields, params)
    P_abs = MinMaxScaling(-P_abs)

    if CheckMax(Fields, Angles, P_abs):
        if CheckSymmetrie(Fields, Angles, P_abs):
            epsilonList.append(non_zero_eps)
            print(f'Found {name}')
        else: print('skipped -- no symmetrie')


def main():

    # Your parameter initialization here
    Angles, Fields, params = GetParams()
    params_ = params[:12]

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

    # Generate all combinations for each key (excluding yy and zz)
    # combinations_per_key = {key: [complex(real, imag) for real in values for imag in values] if key != 'zz' else [0] for key in eps}
    combinations_per_key = {key: [complex(real, imag) for real in values for imag in values] if key not in ['yy', 'zz'] else [0] for key in eps}


    # Create all combinations of the eps dictionary
    all_combinations = [dict(zip(eps.keys(), combination)) for combination in itertools.product(*combinations_per_key.values())]
    # num_combinations = len(all_combinations)
    

        # Prepare parameters for parallel processing
    params_list = [(combination, params_) for combination in all_combinations]

    # Use MPIPoolExecutor to process combinations in parallel
    with MPIPoolExecutor() as executor:
        with tqdm(total=len(all_combinations), desc="Processing", position=0, leave=True) as pbar:
            for _ in executor.map(process_combination, params_list):
                pbar.update(1)

    # Save the result_list to a .txt file
    with open('non_zero_eps_list.txt', 'w') as file:
        for item in epsilonList:
            file.write(f"{item}\n")




if __name__ == "__main__":
    main()