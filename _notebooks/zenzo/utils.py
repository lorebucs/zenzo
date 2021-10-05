import numpy as np
import matplotlib.pyplot as plt

def plot_line(m, b, ax, xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), step=0.1, color='g--'):
    """Plot line with slope ``m`` and intercept ``b``."""
    x = np.arange(ax.get_xlim()[0], ax.get_xlim()[1], step)
    ax.plot(x, m * x + b, color)
     
def plot_2Dpoints(X, y, ax):
    """Plot 2D points for binary classification."""
    class_0 = X[np.argwhere(y==0)]
    class_1 = X[np.argwhere(y==1)]
    ax.scatter([s[0][0] for s in class_0], 
               [s[0][1] for s in class_0], 
               s = 25, 
               color = 'blue', 
               edgecolor = 'k')
    ax.scatter([s[0][0] for s in class_1], 
               [s[0][1] for s in class_1], 
               s = 25, 
               color = 'red', 
               edgecolor = 'k')
      
