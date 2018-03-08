import numpy as np
#from ipywidgets  import FloatSlider, Checkbox, interactive_output, HBox, VBox
from ipywidgets import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

b0 = FloatSlider(min=-10, max=0, step=0.05, value=-4, description = 'b0', 
                    continuous_update=False, layout=Layout(width='275px'))

b1 = FloatSlider(min=0, max=2, step=0.01, value=0.75, description = 'b1',
                    continuous_update=False, layout=Layout(width='275px'))

sl = Checkbox(value=False, description='Show Line', disable=False)
ss = Checkbox(value=False, description='Show Sigmoid', disable=False)
sp = Checkbox(value=False, description='Show Probabilities', disable=False)
sd = Checkbox(value=False, description='Show Decision Boundary',disable=False)

def logistic_regression(b0, b1, sl, ss, sp, sd):

    x = np.array([2, 5, 6, 7, 8, 10])
    y = np.array([0, 0, 1, 0, 1, 1])
    
    def sigmoid(z):
        return 1 / (1 + np.exp(-(b0 + b1*z)))
    
    lm = LinearRegression()
    lm.fit(x.reshape(6,1), y)
    xticks = np.linspace(0, 11, 110)
    y_line = lm.predict(xticks.reshape(110,1))

    plt.close()
    plt.rcParams["figure.figsize"] = [6,4]
    plt.xlim([0,11])
    plt.ylim([-0.2,1.2])
    plt.scatter(x[y==1],y[y==1], c='b')
    plt.scatter(x[y==0],y[y==0], c='r')
    plt.plot([0,11],[0,0], linestyle=':', linewidth=1, c='grey')
    plt.plot([0,11],[1,1], linestyle=':', linewidth=1, c='grey')
    if ss: plt.plot(xticks, sigmoid(xticks))
    if sl: plt.plot(xticks, y_line, c='orange')
    if sp:
        plt.plot([2,2],[sigmoid(2),1],c='r')
        plt.plot([5,5],[sigmoid(5),1],c='r')
        plt.plot([7,7],[sigmoid(7),1],c='r')
        plt.plot([6,6],[0,sigmoid(6)],c='b')
        plt.plot([8,8],[0,sigmoid(8)],c='b')
        plt.plot([10,10],[0,sigmoid(10)],c='b')
    if sd: plt.plot([-b0/b1,-b0/b1],[0,1],linestyle=':', c='orange')    
    plt.show()

def calculation(b0, b1, sl, ss, sp, sd):

    x = np.array([2, 5, 6, 7, 8, 10])
    y = np.array([0, 0, 1, 0, 1, 1])
    
    def sigmoid(z):
        return 1 / (1 + np.exp(-(b0 + b1*z)))
        
    prob = (1-sigmoid(2))*(1-sigmoid(5))*(1-sigmoid(7))*sigmoid(6)*sigmoid(8)*sigmoid(10)
    if sp:
        print('P[Y=0|X=2] * P[Y=0|X=4] * P[Y=1|X=6] * P[Y=0|X=7] * P[Y=1|X=8] * P[Y=1|X=10]')
        print('    =', round(1-sigmoid(2),4),'*',round(1-sigmoid(4),4),'*',
              round(sigmoid(6),4),'*',round(1-sigmoid(7),4),'*',
              round(sigmoid(8),4),'*',round(sigmoid(10),4))
        print('    =', round(prob,5))
    else:
        print()


cdict = {'b0':b0, 'b1':b1, 'sl':sl, 'ss':ss, 'sp':sp, 'sd':sd}

plot_out = interactive_output(logistic_regression, cdict)
calc_out = interactive_output(calculation, cdict)
ui = VBox([b0, b1, sl, ss, sp, sd])

display(HBox([ui, plot_out]))
display(calc_out)
