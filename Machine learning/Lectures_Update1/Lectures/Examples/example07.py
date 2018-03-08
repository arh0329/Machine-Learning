import numpy as np
import pandas as pd
import ipywidgets as widgets
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def poly_regression(y3,sm1, sm3,sm5):


    x1 = np.array([1,2,3,4,5])
    x2 = x1**2
    x3 = x1**3
    x4 = x1**4
    x5 = x1**5
    X = pd.DataFrame({'x1':x1,'x2':x2,'x3':x3,'x4':x4,'x5':x5})
    y = np.array([5, 7, y3, 9, 8])
        
    m1 = LinearRegression()
    m1.fit(X.iloc[:,0:1], y)
    
    m2 = LinearRegression()
    m2.fit(X.iloc[:,0:2], y)

    m3 = LinearRegression()
    m3.fit(X.iloc[:,0:3], y)    
    
    m4 = LinearRegression()
    m4.fit(X.iloc[:,0:4], y)
    
    m5 = LinearRegression()
    m5.fit(X.iloc[:,:], y)
    
    xgrid = np.linspace(0,6,60)
    
    XGrid = pd.DataFrame({'x1':xgrid,
                          'x2':xgrid**2,
                          'x3':xgrid**3,
                          'x4':xgrid**4,
                          'x5':xgrid**5})
    
    
    p1 = m1.predict(XGrid.iloc[:,0:1])
    p3 = m3.predict(XGrid.iloc[:,0:3])
    p5 = m5.predict(XGrid.iloc[:,0:5])
    
    plt.close()
    plt.scatter(x1,y)
    if sm1: plt.plot(xgrid,p1)
    if sm3: plt.plot(xgrid,p3)
    if sm5: plt.plot(xgrid,p5)
    plt.ylim([0,15])
    plt.xlim([0,6])
    plt.show()
    
    if sm1: print("r^2 (deg 1):", m1.score(X.iloc[:,0:1],y))
    if sm3: print("r^2 (deg 3):", m3.score(X.iloc[:,0:3],y))
    if sm5: print("r^2 (deg 5):", m5.score(X.iloc[:,0:5],y))
    
    

_ = widgets.interact(poly_regression,
                     y3=widgets.FloatSlider(min=0,max=15,step=0.1,value=6,continuous_update=False),
                     sm1=widgets.Checkbox(value=False,description='Degree 1',disable=False),
                     sm3=widgets.Checkbox(value=False,description='Degree 3',disable=False),
                     sm5=widgets.Checkbox(value=False,description='Degree 5',disable=False))
    
