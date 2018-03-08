import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model, svm, neighbors, tree

###################################
# Generate Data 
###################################

#sd = np.random.choice(range(0,200))
sd = 194
np.random.seed(sd)
#print("Seed:", sd)

N = 80
x1 = np.random.uniform(0.5,9.5,N)
x2 = np.random.uniform(0.5,9.5,N)

clrs = np.array(['b','r'])

z = 0.05 * (x1**2 + 4*(x2-3)**2 - 30)
prob = 1 / (1 + np.exp(-z))
rolls = np.random.uniform(0,1,N)
y = (prob > rolls).astype(int)

###################################
# Generate Scatterplot
###################################

X = pd.DataFrame({'x1':x1, 'x2':x2})

df = pd.DataFrame({'x1':x1, 'x2':x2, 'y':y})
df.to_csv('data.txt', sep='\t')

plt.close()
plt.rcParams["figure.figsize"] = [5,5]
plt.scatter(x1,x2,c=clrs[y])
plt.show()

###################################
# Create grid for heatmap
###################################

xTicks = np.linspace(0, 10, 1000)
yTicks = np.linspace(0, 10, 1000)
xGrid, yGrid = np.meshgrid(xTicks, yTicks)
grid = np.vstack((xGrid.flatten(), yGrid.flatten())).T

###################################
# Prepare plots
###################################

plt.close()
plt.rcParams["figure.figsize"] = [15,10]
plt.figure()


###################################
# Logistic Regression
###################################

mod01 = linear_model.LogisticRegression(C = 1000000)
mod01.fit(X,y)

prob = mod01.predict_proba(X=grid)
sel = (prob[:,1] > 0.5).astype(int)
selGrid = sel.reshape(xGrid.shape)

myCmap = matplotlib.colors.ListedColormap(['#AAAAFF', '#FFAAAA'])
plt.subplot(2,3,1)
plt.pcolormesh(xTicks, yTicks, selGrid, cmap=myCmap)
plt.scatter(x1,x2,c=clrs[y])
plt.title('Logistic Regression')

###################################
# SVM, RBF Kernel
###################################

mod02 = svm.SVC(kernel='rbf', gamma=0.9, C=1.0)
mod02.fit(X,y)

pred = mod02.predict(X=grid)
predGrid = pred.reshape(xGrid.shape)

myCmap = matplotlib.colors.ListedColormap(['#AAAAFF', '#FFAAAA'])
plt.subplot(2,3,2)
plt.pcolormesh(xTicks, yTicks, predGrid, cmap=myCmap)
plt.scatter(x1,x2,c=clrs[y])
plt.title('SVM (RBF Kernel')

###################################
# SVM, Polynomial Kernel
###################################

mod03 = svm.SVC(kernel='poly', degree=3, C=1.0)
mod03.fit(X,y)

pred = mod03.predict(X=grid)
predGrid = pred.reshape(xGrid.shape)

myCmap = matplotlib.colors.ListedColormap(['#AAAAFF', '#FFAAAA'])
plt.subplot(2,3,3)
plt.pcolormesh(xTicks, yTicks, predGrid, cmap=myCmap)
plt.scatter(x1,x2,c=clrs[y])
plt.title('SVM (Polynomial Kernel)')

###################################
# KNN, N=1
###################################

mod04 = neighbors.KNeighborsClassifier(n_neighbors=1)
mod04.fit(X,y)

pred = mod04.predict(X=grid)
predGrid = pred.reshape(xGrid.shape)

myCmap = matplotlib.colors.ListedColormap(['#AAAAFF', '#FFAAAA'])
plt.subplot(2,3,4)
plt.pcolormesh(xTicks, yTicks, predGrid, cmap=myCmap)
plt.scatter(x1,x2,c=clrs[y])
plt.title('KNN (K=1)')

###################################
# KNN, N=3
###################################

mod04 = neighbors.KNeighborsClassifier(n_neighbors=3)
mod04.fit(X,y)

pred = mod04.predict(X=grid)
predGrid = pred.reshape(xGrid.shape)

myCmap = matplotlib.colors.ListedColormap(['#AAAAFF', '#FFAAAA'])
plt.subplot(2,3,5)
plt.pcolormesh(xTicks, yTicks, predGrid, cmap=myCmap)
plt.scatter(x1,x2,c=clrs[y])
plt.title('KNN (K=3)')



###################################
# Decision Tree
###################################

mod05 = tree.DecisionTreeClassifier()
mod05.fit(X,y)

pred = mod05.predict(X=grid)
predGrid = pred.reshape(xGrid.shape)

myCmap = matplotlib.colors.ListedColormap(['#AAAAFF', '#FFAAAA'])
plt.subplot(2,3,6)
plt.pcolormesh(xTicks, yTicks, predGrid, cmap=myCmap)
plt.scatter(x1,x2,c=clrs[y])
plt.title('Decision Tree')

plt.show()



