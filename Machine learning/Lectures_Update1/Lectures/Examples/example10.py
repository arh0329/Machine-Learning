import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import sklearn.datasets as skds
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

X, y = skds.make_classification(n_samples = 200, n_classes = 2,
                                n_features=2, n_redundant=0, 
                                n_informative=2, n_clusters_per_class=2,) 


minx, maxx = min(X[:,0]), max(X[:,0])
marginx = (maxx - minx) / 20
x0, x1 = minx - marginx, maxx + marginx
    
miny, maxy = min(X[:,1]), max(X[:,1])
marginy = (maxy - miny) / 20
y0, y1 = miny - marginy, maxy + marginy
    
xticks = np.linspace(x0, x1, 500)
yticks = np.linspace(y0, y1, 500)

xGrid, yGrid = np.meshgrid(xticks, yticks)
grid = np.vstack((xGrid.flatten(), yGrid.flatten())).T
myCmap = matplotlib.colors.ListedColormap(['#AAAAFF', '#FFAAAA'])

    
plt.close()
plt.rcParams["figure.figsize"] = [12,12]                                       



############################################
# KNN (K = 3)
############################################
                                           
mod02 = KNeighborsClassifier(3)
mod02.fit(X,y)
acc = round(100*mod02.score(X,y),2)

pred = mod02.predict(X=grid)
predGrid = pred.reshape(xGrid.shape)

plt.subplot(2,2,1)
plt.pcolormesh(xticks, yticks, predGrid, cmap=myCmap)
plt.scatter(X[y==0,0],X[y==0,1],c='b')
plt.scatter(X[y==1,0],X[y==1,1],c='r')
plt.axis('off')
plt.title('KNN (K=3)' + str(acc) + '%')

############################################
# KNN (K = 5)
############################################
                                           
mod03 = KNeighborsClassifier(7)
mod03.fit(X,y)
acc = round(100*mod03.score(X,y),2)

pred = mod03.predict(X=grid)
predGrid = pred.reshape(xGrid.shape)

plt.subplot(2,2,2)
plt.pcolormesh(xticks, yticks, predGrid, cmap=myCmap)
plt.scatter(X[y==0,0],X[y==0,1],c='b')
plt.scatter(X[y==1,0],X[y==1,1],c='r')
plt.axis('off')
plt.title('KNN (K=5)' + str(acc) + '%')

############################################
# KNN (K = 7)
############################################
                                           
mod04 = KNeighborsClassifier(7)
mod04.fit(X,y)
acc = round(100*mod04.score(X,y),2)

pred = mod04.predict(X=grid)
predGrid = pred.reshape(xGrid.shape)

plt.subplot(2,2,3)
plt.pcolormesh(xticks, yticks, predGrid, cmap=myCmap)
plt.scatter(X[y==0,0],X[y==0,1],c='b')
plt.scatter(X[y==1,0],X[y==1,1],c='r')
plt.axis('off')
plt.title('KNN (K=7)' + str(acc) + '%')

############################################
# Logistic Regression
############################################
                                           
mod01 = LogisticRegression(C=1e90)
mod01.fit(X,y)
acc = round(100*mod01.score(X,y),2)

pred = mod01.predict(X=grid)
predGrid = pred.reshape(xGrid.shape)

plt.subplot(2,2,4)
plt.pcolormesh(xticks, yticks, predGrid, cmap=myCmap)
plt.scatter(X[y==0,0],X[y==0,1],c='b')
plt.scatter(X[y==1,0],X[y==1,1],c='r')
plt.axis('off')
plt.title('Logistic Regression -- ' + str(acc) + '%')



plt.show()

