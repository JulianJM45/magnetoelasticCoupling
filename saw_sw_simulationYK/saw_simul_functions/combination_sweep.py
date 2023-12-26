from saw_simul_functions import *
import time
import itertools


def CombinationSweep(Angles, Fields, params):

    alpha, AniType, mue0Hani, phiu, A, g, mue0Ms, k, b1, b2, f, t = params

    eps = {
    'xx': 1,
    'yy': 0,
    'zz': 0,
    'xy': 0,
    'xz': 0,
    'yz': 0
    }

    # Values for real and imaginary parts to sweep through
    values = [-1, -0.5, 0, 0.5, 1]

    # Generate all combinations for each key (excluding zz)
    combinations_per_key = {key: [complex(real, imag) for real in values for imag in values] if key != 'zz' else [0] for key in eps}

    # Create all combinations of the eps dictionary
    all_combinations = [dict(zip(eps.keys(), combination)) for combination in itertools.product(*combinations_per_key.values())]

    # Print the total number of combinations
    print("Total number of combinations:", len(all_combinations))
    notify = int(len(all_combinations)/10000)

    # Integer indices based on boolean indices
    pos_angle_indices = np.where(Angles > 0)[0]
    neg_angle_indices = np.where(Angles < 0)[0]
    pos_field_indices = np.where(Fields > 0)[0]
    neg_field_indices = np.where(Fields < 0)[0]

    angle_interval = range(10, 60)
    field_interval = range(15, 40)

    

    start_time = time.time()
    # Print the first few combinations (optional)
    for i, combination in enumerate(all_combinations):
        eps = combination
        params = [alpha, AniType, mue0Hani, phiu, A, g, mue0Ms, k, eps['xx'], eps['yy'], eps['yy'], eps['xy'], eps['xz'], eps['yz'], b1, b2, f, t]
        P_abs = calculate(Angles, Fields, params)
        non_zero_eps = {key: value for key, value in eps.items() if value != 0}
        name = f'Ray+{non_zero_eps}'

        min_indices = np.unravel_index(np.argmin(P_abs), P_abs.shape)
        # Check if indices are in the specified intervals or their negatives
        is_within_intervals = (
            (min_indices[0] in angle_interval and -min_indices[1] in field_interval) or
            (-min_indices[0] in angle_interval and min_indices[1] in field_interval)
        )
            
        if is_within_intervals:
            #filter 2
            sum_of_positive_P_abs = np.sum(P_abs[:, pos_field_indices])    
            sum_of_negative_P_abs = np.sum(P_abs[:, neg_field_indices])
            difference1 = np.abs(sum_of_positive_P_abs-sum_of_negative_P_abs)

            if difference1 < 1e18:
                #filter 3
                sum_of_positive_P_abs = np.sum(P_abs[pos_angle_indices[:, np.newaxis], neg_field_indices])    
                sum_of_negative_P_abs = np.sum(P_abs[neg_angle_indices[:, np.newaxis], pos_field_indices])
                difference2 = np.abs(sum_of_positive_P_abs-sum_of_negative_P_abs)

                if difference2 < 1e18:
                    # Plotting
                    Plot(np.rad2deg(Angles), Fields, P_abs, name=name)
                    

                else: print(f'skip {name}')

            else: print(f'skip {name}')

        # else: print(f'skip {name}')
        
        if i % notify == 0:
            # Measure elapsed time
            elapsed_time = time.time() - start_time
            # Calculate average time per iteration
            RT = remaining_time(i, all_combinations, elapsed_time)
            # Print progress information
            print(f"Iteration {round((i + 1)/len(all_combinations)*100, 2)}% - Remaining Time: {RT}")

    # # check symmetrie
    #     sum_of_positive_P_abs = np.sum(P_abs[pos_angle_indices[:, np.newaxis], neg_field_indices])    
    #     sum_of_negative_P_abs = np.sum(P_abs[neg_angle_indices[:, np.newaxis], neg_field_indices])
    #     bigger = sum_of_positive_P_abs > sum_of_negative_P_abs


def remaining_time(i, all_combinations, elapsed_time):
        avg_time_per_iteration = elapsed_time / (i + 1)
        # Calculate remaining time
        remaining_iterations = len(all_combinations) - (i + 1)
        remaining_time = avg_time_per_iteration * remaining_iterations

        # Convert remaining time to minutes and hours
        remaining_minutes = remaining_time / 60
        remaining_hours = remaining_time / 3600

        if remaining_hours >= 1:
            hours = int(remaining_hours)
            minutes = int(remaining_hours % hours * 60)
            return (f'{hours}h and {minutes}min')
        else:
            minutes = int(remaining_minutes)
            seconds = int(remaining_minutes % minutes *60)
            return (f'{minutes}min and {seconds}s')