import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    importfolder = r'C:\Users\Julian\Documents\BA\FMR#3'
    filename = 'DerivateDevideM05.dat'

    importfilepath = os.path.join(importfolder, filename)

    column_names = ['Field', 'Frequency', 'S21']

    df = pd.read_csv(importfilepath, delimiter='\t', skiprows=4, names=column_names)

    # Extract columns as separate arrays
    Fields = df['Field'].to_numpy()
    Frequencies = df['Frequency'].to_numpy()
    S21 = df['S21'].to_numpy()

    LengthOneAngle = int(len(Fields)/9)

    #update for first angle
    Fields = Fields[:LengthOneAngle]*-1000
    Frequencies = Frequencies[:LengthOneAngle]*1e-9
    S21 = S21[:LengthOneAngle]

    X, Y, Z = CreateMatrix(Fields, Frequencies, S21)

    cmPlot(Z, X, Y, name='FMR', save=True)


def cmPlot(Z, X, Y, name='Rayleigh', outputfolder=r'C:\Users\Julian\Pictures\BA', show=True, save=False, cmap='seismic'):

    # Set the desired figure size (width_cm, height_cm) in centimeters
    width_cm = 16  # Adjust width as needed in cm
    height_cm = 9  # Adjust height as needed in cm
    fontsize =11

    # Convert centimeters to inches
    width_in = width_cm / 2.54
    height_in = height_cm / 2.54


    # Create the figure with the specified size
    fig = plt.figure(figsize=(width_in, height_in), dpi=300)


    # vmin=np.min(Z)
    # vmax=np.max(Z)
    vmax_abs = np.max(np.abs(Z))
    vmin = -vmax_abs
    vmax = vmax_abs

    # print(vmin, vmax)
    # vmax = 0
    pcm = plt.pcolormesh(
        X, Y, Z,
        cmap=cmap,
        vmin=vmin,  # Set the minimum value for the color scale
        vmax=vmax   # Set the maximum value for the color scale
    )




    cbar = plt.colorbar(pcm)
    cbar.ax.tick_params(labelsize=fontsize)
    # Add labels and a title
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = 'Arial'
    cbar.ax.set_title('Re ($dS_{21}/dH$)', fontsize=fontsize)
    plt.xlabel('$\mu_0H$ (mT)', fontsize=fontsize)
    plt.ylabel('$f$ (GHz)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)  
    plt.yticks(fontsize=fontsize)
    plt.tick_params(axis='both', direction='in', top = True, right = True, width=1, length=4)

    # Save the plot as an image file (e.g., PNG)
    if save: 
        output_filepath = os.path.join(outputfolder, f'ColorMap{name}.pdf')
        plt.savefig(output_filepath, dpi=300, bbox_inches='tight')

    # Show the final plot with all heatmaps
    if show: plt.show()

    plt.clf()

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

if __name__ == "__main__":
    # This block will be executed only if the script is run directly, not when imported as a module
    main()