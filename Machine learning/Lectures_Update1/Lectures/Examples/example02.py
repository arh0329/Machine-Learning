import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt

#sd = np.random.choice(range(0,200))
sd = 196
np.random.seed(sd)
#print("Seed:", sd)
N = 50
x1, x2 = np.random.normal(3,1.5,N), np.random.normal(5,1.5,N)
y1, y2 = np.random.normal(5,1.5,N), np.random.normal(3,1.5,N)
x, y = np.append(x1,x2), np.append(y1,y2)
c = [0]*N + [1]*N
clrs = np.array(['b','r'])
    
def linSepExample(b,m):
    plt.rcParams["figure.figsize"] = [6,6]
    plt.scatter(x,y,c=clrs[c],alpha=0.8)    
    plt.plot([0,8],[b,8*m+b])
    plt.fill([0,8,8,0],[b,8*m+b,8,8],'b',alpha=0.2 )
    plt.fill([0,8,8,0],[b,8*m+b,0,0],'r',alpha=0.2 )
    plt.axis((0,8,0,8))
    plt.show()
    
    boundary = m*x + b
    ptAbove = y > boundary
        
    tBlue = sum( (clrs[c] == 'b') & ptAbove )
    fBlue = sum( (clrs[c] == 'r') & ptAbove )
    tRed = sum( (clrs[c] == 'r') & ~ptAbove )
    fRed = sum( (clrs[c] == 'b') & ~ptAbove )
        
    print("True Blue: ", tBlue)
    print("False Blue:", fBlue)
    print("True Red:  ", tRed)
    print("False Red: ", fRed, "\n")
        
    print("Accuracy:", (tBlue + tRed) / (tBlue + fBlue + tRed + fRed))
        
    
widgets.interact(linSepExample,
                 b=widgets.FloatSlider(min=-4,max=9,step=0.1,value=6,
                                       continuous_update=False),
                 m=widgets.FloatSlider(min=-2,max=2,step=0.1,value=0,
                                       continuous_update=False))