import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from my_modules import *
import numpy as np


# Define the angle offset
alpha = 50

name = 'Rayleigh'

# Define file paths
if name == 'Rayleigh': input_filepath = r'C:\Users\Julian\Documents\BA\Field_and_Angle_Sweep#3\M06\TimeGatedM06.txt'
if name == 'Sezawa': input_filepath = r'C:\Users\Julian\Documents\BA\Field_and_Angle_Sweep#3\M03\TimeGatedM03.txt'
# input_filepath='./magneto_elastic_coupling/TimeGatedM06.txt'

if name == 'Rayleigh': output_folder = r'C:\Users\Julian\Documents\BA\Field_and_Angle_Sweep#3\M06'
if name == 'Sezawa': output_folder = r'C:\Users\Julian\Documents\BA\Field_and_Angle_Sweep#3\M03'


def main():
    # Angle_column, setField_column, Field_column, S12_column = loadData(input_filepath)
    Angles, setFields, Fields, S12 = loadData(input_filepath)

    unique_angles = np.unique(Angles)
    unique_fields = np.unique(Fields)

    average_s12 = averageS12(Angles, Fields, S12, unique_angles)

    DeltaS12 = deltaS12(Angles, S12, average_s12, unique_angles)

    Angles, Fields = updateColums(Angles, Fields)

    # new_data = np.column_stack((Fields, Angles, DeltaS12))

    # exportData(new_data)

    # ResFields = resonanceFields(new_data, save=False)

    X, Y, Z = CreateMatrix(Fields, Angles, DeltaS12)


    Z = FillMatrix(Z)

    # Z = MinMaxScaling(Z)

    cmPlot(Z, Fields, Angles, name='Rayleigh', show=True, save=True)


    # CheckMax(Fields, Angles, Z)

    # CheckSymmetrie(Fields, Angles, Z)


def loadData(input_filepath):
    # Read the data from the .txt file
    data = np.loadtxt(input_filepath, dtype=float, skiprows=1)

    # Extract the angles
    Angle_column = data[:, 0]
    data = data[(Angle_column > -90) & (Angle_column <= 90)]

    #Extract columns
    setField_column = data[:, 1]
    Angle_column = data[:, 0]
    Field_column = 1000 * (data[:, 2] + data[:, 3]) / 2
    S12_column = data[:, 5]
    real_column = data[:, 7]
    imag_column = data[:, 8]

    return Angle_column, setField_column, Field_column, S12_column


def averageS12(Angle_column, Field_column, S12_column, unique_angles):
    # Calculate average S_12 of 3 min/max field elemnts for each angle
    kth = 3
    average_s12 = {}
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

'''
def getMaxs():
    # Retrieve the corresponding setField and Field values
    max_setField = setField_column[max_s12_index]
    max_Field = Field_column[max_s12_index]
    max_S12 = S12_column[max_s12_index]
    max_angle = Angle_column[max_s12_index]
    print(f"Max S12: {max_S12}, SetField: {max_setField}, Field: {-max_Field}, Angle: {max_angle-alpha}")
'''

def deltaS12(Angles, S12, average_s12, unique_angles):
    # Calculate the Delta S_12 column
    deltaS12_column = np.zeros_like(S12)
    for angle in unique_angles:
        mask = Angles == angle
        delta_S12 = S12[mask] - average_s12[angle]
        # Handle invalid or missing values (e.g., NaN)
        delta_S12[np.isnan(delta_S12)] = 0  # Replace NaN with 0 or another suitable value
        deltaS12_column[mask] = delta_S12

    return -deltaS12_column

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

def exportData(new_data):
    # Create a new data matrix with the updated columns
    header = np.array(["Field in mT", "Angle in °", "Delta S_12"])
    output_filepath = os.path.join(output_folder, f'PostTG_{name}.txt')
    np.savetxt(output_filepath, new_data, header='\t'.join(header), comments='', delimiter='\t', fmt='%.6e')


def resonanceFields(new_data, save=False):
    # Get resonance fields
    unique_angles = np.unique(new_data[:, 1])
    min_fields = []
    for angle in unique_angles:
        angle_data = new_data[new_data[:,1] == angle]
        min_index = np.argmin(angle_data[:, 2])
        min_field = np.abs(angle_data[min_index, 0])
        min_fields.append((angle, min_field))

    min_fields = np.array(min_fields)

    if save: 
        header = np.array(["Angle in °", "Field in mT"])
        output_filepath = os.path.join(output_folder, f'ResFields_{name}.txt')
        np.savetxt(output_filepath, min_fields, header='\t'.join(header), comments='', delimiter='\t', fmt='%.6e')
    
    return min_fields

def CreateMatrix(Fields, Angles, DeltaS12):
    unique_fields = np.unique(Fields)
    unique_angles = np.unique(Angles)
    # Create X and Y grids using numpy.meshgrid
    X, Y = np.meshgrid(unique_fields, unique_angles)

    # Initialize Z with None values to indicate empty fields
    Z = np.empty_like(X)

    # Create index maps for angles and fields
    angle_index_map = {angle: index for index, angle in enumerate(unique_angles)}
    field_index_map = {field: index for index, field in enumerate(unique_fields)}

    # Fill the Z-values with corresponding deltaS12 values using advanced indexing
    for angle, field, deltaS12 in zip(Angles, Fields, DeltaS12):
        angle_index = angle_index_map.get(angle)
        field_index = field_index_map.get(field)
        if angle_index is not None and field_index is not None:
            Z[angle_index, field_index] = deltaS12
    X = np.matrix(X)
    Y = np.matrix(Y)
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
















