import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import AutoMinorLocator
# import platform

# if platform.system() == 'Linux': output_folder = '/home/julian/Seafile/BAJulian/picturesForBA'
# elif platform.system() == 'Windows': output_folder=r'C:\Users\Julian\Seafile\BAJulian\picturesForBA'

output_folder = '../picturesForBA'



class Graph:
    def __init__(self, width_cm=16.5, height_cm=None):
        self.width_cm = width_cm
        if height_cm is None:
            self.height_cm = width_cm * (9/16)
        else: self.height_cm = height_cm

        # Convert centimeters to inches
        width_in = self.width_cm / 2.54
        height_in = self.height_cm / 2.54

        self.fontsize = 11
        self.Polar = False
        self.ColorMap = False


        # Create the figure with the specified size
        fig = plt.figure(figsize=(width_in, height_in), dpi=300)

    def go_polar(self, xlabel='', ylabel=''):
        self.Polar = True
        self.ax = plt.subplot(111, projection='polar')
        
        
    def add_plot(self, x, y, label='', color='red', linewidth=1):
        if self.Polar: 
            # ax = plt.subplot(111, projection='polar')
            self.ax.plot(x, y, label=label, color=color, linewidth=linewidth)
        else: plt.plot(x, y, label=label, color=color)

    def add_scatter(self, x, y, label='', marker='o', color='blue', s=1, zorder=1):
        if self.Polar: 
            # ax = plt.subplot(111, projection='polar')
            self.ax.scatter(x, y, label=label, marker=marker, s=s, color=color, zorder=zorder)
        else:plt.scatter(x, y, label=label, marker=marker, s=s, color=color, zorder=zorder)

    def add_errorbar(self, x, y, xerror=None, yerror=None, label='', marker='o', color='blue', s=1, capsize=3):
        plt.errorbar(x, y, xerr=xerror, yerr=yerror, fmt=marker, label=label, elinewidth=1, markersize=s, color=color, capsize=capsize)

    def add_vline(self, x, y=0, label='', color='black', linestyle='-', linewidth=2):
        plt.axvline(x=x, color=color, linestyle=linestyle, linewidth=linewidth)
        plt.text(x+1, y, label, rotation=0, verticalalignment='bottom', color=color)

    def add_cmPlot(self, Z, X, Y, cbarlabel='$\Delta$$S_{21}$\u2009(dB)', cmap='hot_r', equalBounds=False, vmin=None, vmax=None):
        self.ColorMap = True
        if X.ndim == 1:
            print('X is no matrix')
            unique_angles = np.unique(Y)
            unique_fields = np.unique(X)
            # Create X and Y grids using numpy.meshgrid
            X, Y = np.meshgrid(unique_fields, unique_angles)

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
        # Add a vertical colorbar on the right side of the plot
        cbar = plt.colorbar(pcm)
        cbar.ax.tick_params(labelsize=self.fontsize)
        cbar.ax.set_title(cbarlabel, fontsize=self.fontsize)
        


    def plot_Graph(self, show=True, save=False, legend=False, name='Test1', title=None, xlabel=None, ylabel=None, ymin=None, ymax=None,  xmin=None, xmax=None, outputfolder=output_folder):
        fontsize = self.fontsize

        plt.rcParams['font.family'] = 'sans-serif'
        # plt.rcParams['font.sans-serif'] = 'Arial'
        if not self.Polar:
            plt.xlabel(xlabel, fontsize=fontsize)
            plt.ylabel(ylabel, fontsize=fontsize)
            plt.xticks(fontsize=fontsize)   
            # plt.xticks(np.arange(-90, 90, 30), fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            # Add minor ticks
            # plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
            # plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
            plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, width=1, length=4)
            

            if xmin is not None and xmax is not None:    
                plt.xlim([xmin, xmax])
            
            if ymin is not None and ymax is not None:
                plt.ylim([ymin, ymax])

        if self.Polar:
            ax = self.ax
            pos = ax.get_position()
            new_pos = [pos.x0-0.05, pos.y0 + 0.5*pos.height, pos.width, 0.5*pos.height]
            ax2 = ax.figure.add_axes(new_pos, frame_on=False, xticks=[], ylim=ax.get_ylim(), ylabel=ylabel)
            r_ticks = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num=5)
            ax2.set_yticks(r_ticks)
            ax2.yaxis.set_label_coords(new_pos[0]-0.5, new_pos[1])
            ax.set_xticklabels([f'\u2001\u2001{xlabel} = 0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'], fontsize=fontsize)
            ax.set_yticklabels([])

        # plt.xlim([-10, 850])

        if legend: plt.legend()
            # Set the title with left alignment
        if title is not None:
            plt.title(title, loc='left')

            # Rest of the code...
        # Save the plot as an image file (e.g., PNG)
        if save:
            if self.ColorMap: 
                output_filepath = os.path.join(outputfolder, f'ColorMap{name}.png')
                plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
            else:
                output_filepath = os.path.join(outputfolder, f'{name}.pdf')
                plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
            # plt.tight_layout()
            # plt.savefig(output_filepath, dpi=300)

        # Show the final plot
        
        if show: plt.show()

        plt.clf()

        # plt.clf()




## exmaple usage of graph class
'''
if __name__ == "__main__":
    # Example usage
    graph = Graph()

    x1 = [1, 2, 3, 4]
    y1 = [2, 4, 6, 8]
    graph.add_scatter(x1, y1, label='Plot 1', color='red')

    x2 = [1, 2, 3, 4]
    y2 = [1, 4, 9, 16]
    graph.add_scatter(x2, y2, label='Plot 2', color='blue')

    graph.plot_Graph()
'''





def GraphPlot(x, y, xlabel='', ylabel='', name='Test1', title=None, outputfolder=output_folder, scatter=True, s=1, show=True, save=False, width_cm = 16.5, height_cm = None, color='blue', ymin=None, ymax=None,  xmin=None, xmax=None):
    fontsize = 11

    # Convert centimeters to inches
    width_in = width_cm / 2.54
    if height_cm is None:
        height_cm = width_cm * (9/16)
    height_in = height_cm / 2.54

    # Create the figure with the specified size
    fig = plt.figure(figsize=(width_in, height_in), dpi=300)

    plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = 'Arial'
    if title is not None: plt.title(title)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)  
    plt.yticks(fontsize=fontsize)
    plt.tick_params(axis='both', direction='in', top = True, right = True, width=1, length=4)

    if xmin is not None and xmax is not None:    
        plt.xlim([xmin, xmax])
    
    if ymin is not None and ymax is not None:
        plt.ylim([ymin, ymax])

    

    # Add a legend
    # plt.legend()

    if scatter: plt.scatter(x, y, label=name, marker='x', s=s, color=color)
    else: plt.plot(x, y, label=name, color=color)

        # Save the plot as an image file (e.g., PNG)
    if save: 
        output_filepath = os.path.join(outputfolder, f'{name}.eps')
        plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
        # plt.savefig(output_filepath, dpi=300)

    # Show the final plot with all heatmaps
    if show: plt.show()

    plt.clf()



def get_plot_colors(num_colors, alpha=1):
    colors = plt.cm.rainbow(np.linspace(0, 1, num_colors))
    colors[:, 3] = alpha  # Set alpha value for transparency
    return colors