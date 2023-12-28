import numpy as np

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
        
    angle_interval = range(15, 35)
    field_interval = range(25, 38)
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
    pos_angle_indices = np.where(Angles > 0)
    neg_angle_indices = np.where(Angles < 0)
    
    sum_of_positive_P_abs = np.sum(P_abs[:, pos_field_indices])    
    sum_of_negative_P_abs = np.sum(P_abs[:, neg_field_indices])
    difference1 = np.abs(sum_of_positive_P_abs - sum_of_negative_P_abs)
    # print(difference1)
    if difference1 < 20:
        return True
    else:
        
        return False
        

'''
        if difference1 < 1e18:
            sum_of_positive_P_abs = np.sum(P_abs[pos_angle_indices[:, np.newaxis], neg_field_indices])    
            sum_of_negative_P_abs = np.sum(P_abs[neg_angle_indices[:, np.newaxis], pos_field_indices])
            difference2 = np.abs(sum_of_positive_P_abs - sum_of_negative_P_abs)

            if difference2 < 1e18:
                Plot(np.rad2deg(Angles), Fields, P_abs, name=name)
            else:
                print(f'skip {name}')
        else:
            print(f'skip {name}')
            '''