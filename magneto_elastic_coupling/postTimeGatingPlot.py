import os
from my_modules import *
import numpy as np
from scipy.optimize import curve_fit
from collections import defaultdict


input_folder = '../dataForPython/Field_Angle_Sweep#3'
output_folder = '../dataForPython/Field_Angle_Sweep#3'

# Define the angle offset
alpha = 48
# alpha = 0




name = 'Rayleigh'
# name = 'Sezawa'

def main():
    S12 = loadData()
    # print(S12.keys())

    average_s12 = averageS12(S12)

    DeltaS12 = deltaS12(S12, average_s12)
    
    
    # exportData(DeltaS12)

    PlotOneAngle(name, DeltaS12, angles=[0, 30, 42, 88], save=True)

    # ResFields = resonanceFields(DeltaS12, abs=False, save=False)     #abs=True for calcOffset, abs=False for resonanceS21

    # # # calcOffset(ResFields)

    # ResS21 = resonanceS21(DeltaS12, ResFields, save=True)   

    # calcRes21(ResS21, save=False, sign='pos', angles=[]) 
    
    # X, Y, Z = CreateMatrix(DeltaS12)
    # Z = FillMatrix(Z)
    
    # # Z = MinMaxScaling(Z)

    # cmPlot(Z, X, Y, name=name, vmax=-0.08, cmap='hot', show=True, save=False)


    # CheckMax(Fields, Angles, Z)

    # CheckSymmetrie(Fields, Angles, Z)





def calcRes21(ResS21_dict, save=False, sign='neg', angles=[0]):
    Angles = []
    ResS21 = []
    angle_sums = defaultdict(float)
    angle_counts = defaultdict(int)
    for (field, angle), value in ResS21_dict.items():
        if sign == 'pos' and field > 0:
            Angles.append(angle)
            ResS21.append(value)
            angle_sums[angle] += value
            angle_counts[angle] += 1 
        elif sign == 'neg' and field < 0:
            Angles.append(angle)
            ResS21.append(value)
            angle_sums[angle] += value
            angle_counts[angle] += 1          
    angle_means = {angle: angle_sums[angle] / angle_counts[angle] for angle in angle_sums}
    Angles = np.array(list(angle_means.keys()))
    ResS21 = np.array(list(angle_means.values()))

    max_angle = Angles[np.argmax(ResS21)] #change for positive or negative deltaS12
    print(f"The angle where ResS21 is maximal for {sign}Fiels: {max_angle}")
    min_angle = Angles[np.argmin(ResS21)] #change for positive or negative deltaS12
    print(f"The angle where ResS21 is minimal for {sign}Fiels: {min_angle}")

    
    mygraph = Graph(width_cm=8.2)
    colors = get_plot_colors(len(angles))
    mygraph.add_scatter(Angles, ResS21, s=4)
    for i, angle in enumerate(angles):
        color = colors[i]
        mygraph.add_vline(x=angle, color=color, linewidth=1.3, label=f'{angle}°')
        mygraph.add_vline(x=-angle, color=color, linewidth=1.3, label=f'{-angle}°')    
    
    if sign == 'pos': title=f'b) \t\t $\mu_0H_0>0$'
    elif sign == 'neg': title=f'a) \t\t $\mu_0H_0<0$'
    mygraph.plot_Graph(save=save, legend=False, title=title, xlabel='$\phi_H$\u2009(°)', ylabel='min $\Delta$$S_{21}$\u2009(dB)', name=f'ResS21_{sign}Fields{name}')
    

def resonanceS21(DeltaS12_dict, ResFields, save=False):
    ResS21 = {}
    kth = 8


    if name == 'Rayleigh': x0=34.23
    if name == 'Sezawa': x0=62.82

    # filter FMR resonance fields
    for i in range(5):
        closest_angle = None
        min_diff = float('inf')
        for angle, fieldx in ResFields.items():
            diff = abs(fieldx - x0)
            if diff < min_diff:
                min_diff = diff
                closest_field = fieldx
                closest_angle = angle
        # Delete the closest field from the dictionary
        if closest_angle is not None:
            del ResFields[closest_angle]
        closest_angle = None
        min_diff = float('inf')
        for angle, fieldx in ResFields.items():
            diff = abs(fieldx + x0)
            if diff < min_diff:
                min_diff = diff
                closest_field = fieldx
                closest_angle = angle
        # Delete the closest field from the dictionary
        if closest_angle is not None:
            del ResFields[closest_angle]


    for angle, fieldx in ResFields.items():
        # filter FMR resonance fields
        fields = [field for ((field, angle_in_dict), value) in DeltaS12_dict.items() if angle == angle_in_dict]
        field_index = fields.index(fieldx)
        for i in range((field_index - kth), (field_index + kth + 1)):
            field = fields[i]
            ResS21[(field, angle)] = DeltaS12_dict[(field, angle)]

        # handle opposite resfields
        closest_field = None
        min_diff = float('inf')
        for field in fields:
            diff = abs(field + fieldx)
            if diff < min_diff:
                min_diff = diff
                closest_field = field
        field_index = fields.index(closest_field)
        for i in range((field_index - kth), (field_index + kth + 1)):
            field = fields[i]
            ResS21[(field, angle)] = DeltaS12_dict[(field, angle)]
        

    if save: 
        header = np.array(["Field in mT", "Angle in °", "ResDelta S_12"])
        output_filepath = os.path.join(output_folder, f'ResS21_{name}.txt')
        dtype = [('field', float), ('angle', float), ('deltaS12', float)]
        ResS21_array = np.array([(field, angle, deltaS12) for (field, angle), deltaS12 in ResS21.items()], dtype=dtype)
        deltaS12_column = ResS21_array['deltaS12']
        scaled_deltaS12_column = MinMaxScaling(deltaS12_column)
        ResS21_array['deltaS12'] = scaled_deltaS12_column
        np.savetxt(output_filepath, ResS21_array, header='\t'.join(header), comments='', delimiter='\t', fmt='%.6e')
    
    return ResS21


def calcOffset(ResFields):
    mygraph = Graph()
    Angles = np.array(list(ResFields.keys()))
    Fields = np.array(list(ResFields.values()))
    # Sort the angles and fields
    sorted_indices = np.argsort(Angles)
    Angles = Angles[sorted_indices]
    Fields = Fields[sorted_indices]

    initial_guess = [15, 1, 50, 25]
    params, covariance = curve_fit(cosFunc, Angles, Fields, p0=initial_guess)
    a, b, c, d = params
    fields = cosFunc(Angles, a, b, c, d)
    
    print(f'alpha = {c}')

    mygraph.add_scatter(Angles, Fields)
    mygraph.add_plot(Angles, fields)
    
    mygraph.plot_Graph()


def cosFunc(x, a, b, c, d):
    return a*np.cos(b * np.deg2rad(x-c))+d


def PlotOneAngle(name, DeltaS12_dict, angles=[0], save=False):
    mygraph = Graph()
    colors = get_plot_colors(len(angles))
    if name == 'Rayleigh': x0=34.23
    if name == 'Sezawa': x0=62.82
    # mygraph.add_vline(x=x0, color='yellow')
    # mygraph.add_vline(x=-x0, color='yellow')
    for i, angle in enumerate(angles):
        color = colors[i]
        field_list = []
        deltaS12_list = []
        for (field, angle_in_dict), deltaS12 in DeltaS12_dict.items():
            if angle == angle_in_dict:
                field_list.append(field)
                deltaS12_list.append(deltaS12)
                
        mygraph.add_plot(field_list, deltaS12_list, label=f'{angle}°', linewidth=0.1, color=color)
        # mygraph.add_scatter(field_list, deltaS12_list, label=f'{angle}°', s=4, color=color)
    mygraph.plot_Graph(save=save, legend=True, xlabel='$\mu_0H_0$\u2009(mT)', ylabel='$\Delta S_{21}$\u2009(dB)', name=f'singleAngle_{name}')


def loadData():
    # Define file paths
    filename = 'TimeGated2.45.txt' if name == 'Rayleigh' else 'TimeGated3.53.txt' if name == 'Sezawa' else None
    if filename is None:
        print('no file for this name')
        return

    filepath = os.path.join(input_folder, filename)

    # Read the data from the .txt file
    data = np.loadtxt(filepath, dtype=float, skiprows=1)

    # Filter the data
    Angle_column = data[:, 0]
    data = data[(Angle_column > -90) & (Angle_column <= 90)]
    setField_column = data[:, 1]
    # maxfield = 45e-3 if name == 'Rayleigh' else 80e-3 if name == 'Sezawa' else None
    # if maxfield is None:
    #     print('no fieldborder for this name')
    #     return
    # data = data[(setField_column >= -maxfield) & (setField_column <= maxfield)]

    # Extract and update columns
    Angle_column, Field_column = updateColums(data[:, 0], 1000 * (data[:, 2] + data[:, 3]) / 2)
    S12_column = data[:, 5]

    #  # Get the sorted indices of the angles
    # sorted_indices = np.argsort(Angle_column)

    # # Sort the angles, fields and S12_column
    # Angle_column = Angle_column[sorted_indices]
    # Field_column = Field_column[sorted_indices]
    # S12_column = S12_column[sorted_indices]


    S12 = {}
    for i, row in enumerate(S12_column):
        angle = Angle_column[i]
        field = Field_column[i]
        S12[(field, angle)] = row

    return S12


def averageS12(S21_dict):
    # Extract Angle_column, Field_column, and S12_column from S21_dict
    Angle_column = []
    Field_column = []
    S12_column = []
    for (field, angle), s12 in S21_dict.items():
        Field_column.append(field)
        Angle_column.append(angle)
        S12_column.append(s12)

    # Convert lists to numpy arrays
    Angle_column = np.array(Angle_column)
    Field_column = np.array(Field_column)
    S12_column = np.array(S12_column)

    # Calculate average S_12 of 3 min/max field elements for each angle
    kth = 3
    average_s12 = {}
    unique_angles = np.unique(Angle_column)
    for angle in unique_angles: 
        mask = Angle_column == angle
        smallest_fields = np.partition(Field_column[mask], kth-1)[:kth]
        mask2 = np.isin(Field_column[mask], smallest_fields)
        avg_s12_smallest = np.mean(S12_column[mask][mask2])
        largest_fields = np.partition(Field_column[mask], -kth)[-kth:]
        mask2 = np.isin(Field_column[mask], largest_fields)
        avg_s12_largest = np.mean(S12_column[mask][mask2])
        average_s12[angle] = (avg_s12_smallest+avg_s12_largest)/2

    return average_s12


def deltaS12(S21_dict, average_s12):
    deltaS12_dict = {}
    for (field, angle), s12 in S21_dict.items():
        delta_S12 = s12 - average_s12[angle]
        # deltaS12_dict[(field, angle)] = -delta_S12
        deltaS12_dict[(field, angle)] = delta_S12

    # Field filtering
    maxfield = 45 if name == 'Rayleigh' else 80 if name == 'Sezawa' else None
    if maxfield is None:
        print('no fieldborder for this name')
    deltaS12_dict = {key: value for key, value in deltaS12_dict.items() if -maxfield <= key[0] <= maxfield}

    # fields = set()
    # angles = set()

    # for (field, angle), s12 in deltaS12_dict.items():
    #     fields.add(field)
    #     angles.add(angle)

    # print(f'Number of unique fields: {len(fields)}')
    # print(f'Number of fields per angle: {len(fields) / len(angles)}')
    # print(max(fields))
    # print(f'Number of unique angles: {len(angles)}')

    return deltaS12_dict


def updateColums(Angles, Fields):
    # Find the indices of rows where the "Angle_column" is within the specified range
    Angles = Angles - alpha
    # indices_to_modify = np.where((Angle_column >= -90) & (Angle_column < (-90 + alpha)))
    indices_to_modify = np.where(Angles < (-90))
    Fields[indices_to_modify] = -Fields[indices_to_modify]
    Angles[indices_to_modify] += 180
    # Update columns
    Fields = - Fields
    
    return Angles, Fields


def exportData(DeltaS12_dict):
    # Convert the DeltaS12_dict to a 2D array
    new_data = []
    for (field, angle), deltaS12 in DeltaS12_dict.items():
        new_data.append([field, angle, deltaS12])
    new_data = np.array(new_data)

    # Create a new data matrix with the updated columns
    header = np.array(["Field in mT", "Angle in °", "Delta S_12"])
    output_filepath = os.path.join(output_folder, f'PostTG_{name}.txt')
    np.savetxt(output_filepath, new_data, header='\t'.join(header), comments='', delimiter='\t', fmt='%.6e')


def resonanceFields(DeltaS12_dict, abs=False, save=False):
    # Get resonance fields
    max_fields = {}
    for (field, angle), deltaS12 in DeltaS12_dict.items():
        if angle not in max_fields or deltaS12 < max_fields[angle][1]:  #change if delta S12 is positive or negative
            max_fields[angle] = (field, deltaS12)

    # Extract the field with maximum deltaS12 for each angle
    if abs: max_fields = {angle: np.abs(field) for angle, (field, deltaS12) in max_fields.items()}
    else: max_fields = {angle: field for angle, (field, deltaS12) in max_fields.items()}

    if save: 
        header = np.array(["Angle in °", "Field in mT"])
        output_filepath = os.path.join(output_folder, f'ResFields_{name}.txt')
        dtype = [('angle', float), ('max_field', float)]
        max_fields_array = np.array([(angle, max_field) for angle, max_field in max_fields.items()], dtype=dtype)
        np.savetxt(output_filepath, max_fields_array, header='\t'.join(header), comments='', delimiter='\t', fmt='%.6e')
    
    return max_fields


def CreateMatrix(S12_dict):
    unique_fields = np.array(sorted(set(field for (field, angle) in S12_dict.keys())))
    unique_angles = np.array(sorted(set(angle for (field, angle) in S12_dict.keys())))

    # Create X and Y grids using numpy.meshgrid
    X, Y = np.meshgrid(unique_fields, unique_angles)

    # Initialize Z with None values to indicate empty fields
    Z = np.empty_like(X)
    # Z[:] = np.nan

    for (field, angle), s12 in S12_dict.items():
        field_index = np.where(unique_fields == field)[0][0]
        angle_index = np.where(unique_angles == angle)[0][0]
        Z[angle_index, field_index] = s12

    return X, Y, Z


def FillMatrix(Z):
    # Fill empty cells of Z
    for row_index in range(Z.shape[0]):
        row = Z[row_index, :]
        non_empty_indices = np.nonzero(row)[0]
        if non_empty_indices.size > 0:
            for i in range(Z.shape[1]):
                if row[i] == 0:
                    nearest_non_empty_index = non_empty_indices[np.argmin(np.abs(non_empty_indices - i))]
                    Z[row_index, i] = row[nearest_non_empty_index]

    return Z







    

if __name__ == "__main__":
    # This block will be executed only if the script is run directly, not when imported as a module
    main()
















