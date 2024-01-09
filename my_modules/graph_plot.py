import matplotlib.pyplot as plt
import numpy as np
import os

output_folder = '/home/julian/BA/pictures'
output_folder=r'C:\Users\Julian\Pictures\BA'

def GraphPlot(x, y, xlabel='', ylabel='', name='Test1', title=None, outputfolder=output_folder, scatter=True, s=1, show=True, save=False, width_cm = 16, height_cm = None, color='blue', ymin=None, ymax=None,  xmin=None, xmax=None):
    fontsize = 11

    # Convert centimeters to inches
    width_in = width_cm / 2.54
    if height_cm is None:
        height_cm = width_cm * (9/16)
    height_in = height_cm / 2.54

    # Create the figure with the specified size
    fig = plt.figure(figsize=(width_in, height_in), dpi=300)

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = 'Arial'
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
        output_filepath = os.path.join(outputfolder, f'{name}.pdf')
        plt.savefig(output_filepath, dpi=300, bbox_inches='tight')

    # Show the final plot with all heatmaps
    if show: plt.show()

    plt.clf()



def get_plot_colors(num_colors):
    colors = plt.cm.rainbow(np.linspace(0, 1, num_colors))
    return colors


class Graph:
    def __init__(self, width_cm=16, height_cm=None):
        self.width_cm = width_cm
        if height_cm is None:
            self.height_cm = width_cm * (9/16)
        else: self.height_cm = height_cm

        # Convert centimeters to inches
        width_in = self.width_cm / 2.54
        height_in = self.height_cm / 2.54

        # Create the figure with the specified size
        fig = plt.figure(figsize=(width_in, height_in), dpi=300)

    def add_plot(self, x, y, label='', color='red'):
        plt.plot(x, y, label=label, color=color)

    def add_scatter(self, x, y, label='', marker='x', color='blue'):
        plt.scatter(x, y, label=label, marker=marker, s=1, color=color)

    def add_vline(self, x, label='', linestyle='--', color='red'):
        plt.axvline(x=x, color=color, linestyle=linestyle, linewidth=2, label=label)

    def plot_Graph(self, show=True, save=False, legend=False, name='Test1', xlabel='', ylabel='', ymin=None, ymax=None,  xmin=None, xmax=None, outputfolder=output_folder):
        fontsize = 11

        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Arial'
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.tick_params(axis='both', direction='in', top=True, right=True, width=1, length=4)

        if xmin is not None and xmax is not None:    
            plt.xlim([xmin, xmax])
        
        if ymin is not None and ymax is not None:
            plt.ylim([ymin, ymax])


        if legend: plt.legend()

        # Save the plot as an image file (e.g., PNG)
        if save:
            output_filepath = os.path.join(outputfolder, f'{name}.pdf')
            plt.savefig(output_filepath, dpi=300, bbox_inches='tight')

        # Show the final plot
        
        if show: plt.show()

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