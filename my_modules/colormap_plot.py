import numpy as np
# import matplotlib
# matplotlib.use('pdf')
import matplotlib.pyplot as plt
import os

# matplotlib.use('TkAgg')

def cmPlot(Z, X, Y, ResFields=None, name='Rayleigh', outputfolder=r'C:\Users\Julian\Pictures\BA', show=True, save=False, cmap='hot_r', width_cm = 16, height_cm = 9):
    if isinstance(X, np.ndarray):
        unique_angles = np.unique(Y)
        unique_fields = np.unique(X)
        # Create X and Y grids using numpy.meshgrid
        X, Y = np.meshgrid(unique_fields, unique_angles)
    # Set the desired figure size (width_cm, height_cm) in centimeters
    fontsize =11

    # Convert centimeters to inches
    width_in = width_cm / 2.54
    height_in = height_cm / 2.54

    # Create the figure with the specified size
    fig = plt.figure(figsize=(width_in, height_in), dpi=300)


    vmin=np.min(Z)
    vmax=np.max(Z)
    # vmax_abs = np.max(np.abs(Z))
    # vmin = -vmax_abs
    # vmax = vmax_abs
    # vmax = 0
    pcm = plt.pcolormesh(
        X, Y, Z,
        cmap=cmap,
        vmin=vmin,  # Set the minimum value for the color scale
        vmax=vmax   # Set the maximum value for the color scale
    )
    if ResFields is not None:
        plt.scatter(ResFields[:, 1], ResFields[:, 0], color='blue', marker='x', label='Resonance Fields')
        plt.scatter(-ResFields[:, 1], ResFields[:, 0], color='blue', marker='x', label='Resonance Fields')


    # Add a vertical colorbar on the right side of the plot
    cbar = plt.colorbar(pcm)
    cbar.ax.tick_params(labelsize=fontsize)
    # Add labels and a title
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = 'Arial'
    cbar.ax.set_title('$\Delta$$S_{21}$ (dB)', fontsize=fontsize)
    plt.xlabel('Field $\mu_0H$ (mT)', fontsize=fontsize)
    plt.ylabel('$θ_H$ (°)', fontsize=fontsize)
    plt.title('Colormap of $\Delta$$S_{12}$ '+f'for {name} mode', fontsize=fontsize)
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


