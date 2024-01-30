import numpy as np
# import matplotlib
# matplotlib.use('pdf')
import matplotlib.pyplot as plt
import os

output_folder = '../picturesForBA'

def cmPlot(Z, X, Y, ResFields=None, name='Rayleigh', xlabel='$\mu_0H_0$\u2009(mT)', ylabel='$\phi_H$\u2009(°)', cbarlabel='$\Delta$$S_{21}$\u2009(dB)', outputfolder=output_folder, show=True, save=False, saveSVG=False, cmap='hot_r', width_cm = 16.5, height_cm = None, equalBounds=False, vmin=None, vmax=None):
    if X.ndim == 1:
        print('X is no matrix')
        unique_angles = np.unique(Y)
        unique_fields = np.unique(X)
        # Create X and Y grids using numpy.meshgrid
        X, Y = np.meshgrid(unique_fields, unique_angles)
    # Set the desired figure size (width_cm, height_cm) in centimeters
    fontsize =11

    if height_cm is None:
        height_cm = width_cm * (9/16)
    # Convert centimeters to inches
    width_in = width_cm / 2.54
    height_in = height_cm / 2.54

    # Create the figure with the specified size
    fig = plt.figure(figsize=(width_in, height_in), dpi=300)

    if vmax is None:
        vmax=np.max(Z)
    if vmin is None:
        vmin=np.min(Z)
    if equalBounds:
        vmax_abs = np.max(np.abs(Z))
        vmin = -vmax_abs
        vmax = vmax_abs

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
    plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = 'Arial'
    cbar.ax.set_title(cbarlabel, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    # plt.title('Colormap of $\Delta$$S_{12}$ '+f'for {name} mode', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)  
    plt.yticks(fontsize=fontsize)
    plt.tick_params(axis='both', direction='in', top = True, right = True, width=1, length=4)

    # Save the plot as an image file (e.g., PNG)
    if save: 
        output_filepath = os.path.join(outputfolder, f'ColorMap{name}.png')
        plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
        # plt.savefig(output_filepath, dpi=300)

    if saveSVG:
        output_filepath = os.path.join(outputfolder, f'ColorMap{name}.svg')
        plt.savefig(output_filepath, dpi=300, bbox_inches='tight')

    # Show the final plot with all heatmaps
    if show: plt.show()

    plt.clf()



