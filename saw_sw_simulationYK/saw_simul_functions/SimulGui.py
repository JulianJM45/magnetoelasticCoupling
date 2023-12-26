import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from calculate import calculate
from Plot import plot

class SAWSimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SAW Simulation GUI")

        # Initialize eps values
        self.eps_values = eps

        # Create GUI components
        self.create_gui()

    def create_gui(self):
        # Frame for GUI components
        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Labels and Entry for each eps value
        row_counter = 0
        for key in self.eps_values.keys():
            ttk.Label(frame, text=f"{key}:").grid(row=row_counter, column=0, sticky=tk.E)
            entry_var = tk.StringVar()
            entry_var.set(str(self.eps_values[key]))
            ttk.Entry(frame, textvariable=entry_var).grid(row=row_counter, column=1, sticky=tk.W)
            row_counter += 1

        # Button to update plot
        ttk.Button(frame, text="Update Plot", command=self.update_plot).grid(row=row_counter, columnspan=2)

        # Matplotlib Figure and Canvas
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.plot_canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.plot_canvas.get_tk_widget().grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Initial plot
        self.update_plot()

    def update_plot(self):
        # Get eps values from entry widgets
        for key in self.eps_values.keys():
            entry_value = self.root.children["frame"].children[key.lower()].get()
            self.eps_values[key] = complex(entry_value)

        # Update plot
        P_abs = calculate(Angles, Fields, params)
        non_zero_eps = {key: value for key, value in self.eps_values.items() if value != 0}
        name = f'Ray+{non_zero_eps}'
        plot(np.rad2deg(Angles), Fields, P_abs, name=name, figure=self.figure)
        self.plot_canvas.draw()

if __name__ == "__main__":
    # Set up initial parameters
    Angles = np.deg2rad(np.linspace(-90, 90, num=91))
    Fields = np.linspace(-0.05, 0.05, num=101)

    # Sweep Definition
    All = len(Angles) * len(Fields)

    # Angles = [np.deg2rad(40)]
    # Fields = [25*0.001]

    # Constants
    hbar = 1.05457173e-34
    mue0 = 4 * np.pi * 1e-7
    mueB = 9.27400968e-24

    # Magn Material Parameters
    alpha = 0.008
    AniType = 0     # 0 = Uniaxial; 1 = cubic; 2 = threefold; 3 = sixfold
    mue0Hani = 0.00012035728532673945
    phiu = 0
    A = 3.740757032734014e-12
    g = 2.0468225676228067
    mue0Ms = 0.17913547682653475
    b1 = 4.37
    b2 = 8.75

    C = 1 / 2000

    # Simulated SAW
    f = 2.43e9
    k =  4668814.253017993

    # Structure Properties
    t = 100e-9

     # Initialize the eps dictionary
    eps = {
        'xx': 1 + 1j,
        'yy': 0,
        'zz': 0,
        'xy': 0,
        'xz': 0,
        'yz': 0
    }

    params = [alpha, AniType, mue0Hani, phiu, A, g, mue0Ms, k, eps['xx'], eps['yy'], eps['yy'], eps['xy'], eps['xz'], eps['yz'], b1, b2, f, t]

    # Initialize the Tkinter app
    root = tk.Tk()
    gui = SAWSimulationGUI(root)
    root.mainloop()
