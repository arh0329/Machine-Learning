import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt

#sd = np.random.choice(range(0,200))
sd = 164
np.random.seed(sd)
#print("Seed:", sd)
N = 12
x = np.random.uniform(low=0,high=10,size=N)

y = 5 + 1.4 * x + np.random.normal(0,2.5,N)

def regressionExample(b,m,e):
    
    yhat = b + m * x
    
    plt.rcParams["figure.figsize"] = [6,6]
    plt.plot([0,10],[b,10*m+b],c='purple')
    
    if e:
        for i in range(len(x)):
            plt.plot([x[i],x[i]],[y[i],yhat[i]],c='black',lw=0.75,zorder=1)
    
    plt.scatter(x,y,zorder=2)        

    plt.axis((0,10,0,25))
    plt.show()
    
    SSE = np.sum((y - yhat)**2)
    
    #print("True Blue: ", tBlue)
    #print("False Blue:", fBlue)
    #print("True Red:  ", tRed)
    #print("False Red: ", fRed, "\n")
    
    print("Sum of Squared Errors:", SSE)
    

_ = widgets.interact(regressionExample,
                     b=widgets.FloatSlider(min=-2,max=10,step=0.1,value=2,continuous_update=False),
                     m=widgets.FloatSlider(min=-2,max=2,step=0.01,value=0,continuous_update=False),
                     e=widgets.Checkbox(value=False,description='Show Errors',disable=False))
