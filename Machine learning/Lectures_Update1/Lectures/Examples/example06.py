from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import pandas as pd

np.random.seed(186)

n = 20
x1 = np.random.uniform(0,10,n)
x2 = np.random.uniform(0,10,n)
y = 7 + 1.3 * x1 + 2.5 * x2 + np.random.uniform(0,12,n)

X = pd.DataFrame({'x1':x1,'x2':x2})

lm02 = LinearRegression()
lm02.fit(X,y)

X = pd.DataFrame({'x1':x1,'x2':x2})

plt.close()
fig = plt.figure()
ax = fig.gca(projection='3d')

x1grid = np.arange(0, 10, 0.25)
x2grid = np.arange(0, 10, 0.25)
x1grid, x2grid = np.meshgrid(x1grid, x2grid)
ygrid = lm02.intercept_ + lm02.coef_[0] * x1grid + lm02.coef_[1] * x2grid
ax.scatter(x1, x2, y, marker = 'o')
ax.plot_surface(x1grid, x2grid, ygrid, cmap=cm.coolwarm, alpha=0.8)

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.show()
