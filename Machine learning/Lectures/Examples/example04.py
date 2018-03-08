import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

#sd = np.random.choice(range(0,200))
sd = 108
np.random.seed(sd)
#print("Seed:", sd)

X, y_true = make_blobs(n_samples=200, centers=4,
                       cluster_std=1.60, random_state=sd)

plt.close()
plt.rcParams["figure.figsize"] = [6,6]
plt.scatter(X[:,0], X[:,1],s=20)
plt.show()

#plt.scatter(X[:, 0], X[:, 1], s=50);

kmeans3 = KMeans(n_clusters=3)
kmeans3.fit(X)
y_kmeans3 = kmeans3.predict(X)

kmeans4 = KMeans(n_clusters=4)
kmeans4.fit(X)
y_kmeans4 = kmeans4.predict(X)

kmeans5 = KMeans(n_clusters=5)
kmeans5.fit(X)
y_kmeans5 = kmeans5.predict(X)


plt.close()
plt.rcParams["figure.figsize"] = [15,5]
plt.figure()

plt.subplot(1,3,1)
plt.scatter(X[:,0], X[:,1],c=y_kmeans3, s=20)
plt.title('K-Means (K = 4)')

plt.subplot(1,3,2)
plt.scatter(X[:,0], X[:,1],c=y_kmeans4, s=20)
plt.title('K-Means (K = 5)')

plt.subplot(1,3,3)
plt.scatter(X[:,0], X[:,1],c=y_kmeans5, s=20)
plt.title('K-Means (K = 6)')

plt.show()

#plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
#centers = kmeans.cluster_centers_
#plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);