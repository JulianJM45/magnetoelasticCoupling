import numpy as np
from my_modules.graph_plot import GraphPlot 

x = np.linspace(0, 10, 100)
y = np.sin(x)

GraphPlot(x, y, name='Test1', xlabel='x (m), x\u2009(m)', ylabel='y', save=False, show=True)


