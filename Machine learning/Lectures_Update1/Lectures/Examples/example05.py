import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

np.random.seed(186)

n = 20
x1 = np.random.uniform(0,10,n)
x2 = np.random.uniform(0,10,n)
y = 7 + 1.3 * x1 + 2.5 * x2 + np.random.uniform(0,12,n)

X = pd.DataFrame({'x1':x1,'x2':x2})

plt.close()
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.scatter(x1, x2, y, marker = 'o')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.show()
