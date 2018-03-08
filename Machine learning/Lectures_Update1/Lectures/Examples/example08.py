import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
def sigmoid_plot(b0, b1):

    x = np.linspace(-10,10,100)
    y = 1 / (1 + np.exp(-b0 - b1 * x))

    
    
    plt.close()
    plt.rcParams["figure.figsize"] = [6,4]
    plt.plot(x,y)
    plt.plot([-10,10],[1,1], linestyle=':', c="r")
    plt.plot([-10,10],[0,0], linestyle=':', c="r")
    plt.plot([0,0],[0,1], linewidth=1, c="dimgray")
    plt.show()   

_ = widgets.interact(sigmoid_plot,
                     b0=widgets.FloatSlider(min=-5,max=5,step=0.1,value=0,
                                               continuous_update=False),
                     b1=widgets.FloatSlider(min=-5,max=5,step=0.1,value=1,
                                               continuous_update=False))