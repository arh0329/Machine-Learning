#Aaron Henningfeld
#P04

import numpy as np 
import pandas as pd
#from ClassificationPlotter import plot_regions

class KNNClassifier:

    def __init__(self, X, y, K):
        
        #convert to numpy arrays
        self.X = np.array(X)
        self.y = np.array(y)
        self.K = K
        self.n_observations = len(X) #find num of observations
        self.classes = np.unique(self.y)# find the classes
        
    def predict(self, X):
        
        predictions = []
        
        
        for row in X:
            
            distances = ((row[0]-X[:,0])**2+(row[1]-X[:,1])**2)

            idx = np.argsort(distances)[:self.K]

            knn_labels = self.y[idx]

            knn_distances = distances[idx]

            best_dist = 0
            best_class = 0
            best_count = 0
        
            for each in self.classes:
                
                temp_count = np.sum(knn_labels==each)

                temp_dist = np.sum(knn_distances[knn_labels==each])

                if temp_count>best_count:
                    best_dist = temp_dist
                    best_class = each
                    best_count = temp_count
                    
                if temp_count==best_count and temp_dist<best_dist:
                    best_dist = temp_dist
                    best_class = each
                    best_count = temp_count
                
            predictions.append(best_class)

        predictions = np.array(predictions)
        return predictions
        

    def score(self, X, y):
        
        y=np.array(y) #pass into numpy array
        y_pred = self.predict(X) #make predictions
        numEqual = np.sum(y == y_pred) #find num of right predictions
        accuracy = numEqual/len(y) #find accuracy
        return accuracy      

        
    def confusion_matrix(self, X,y):
    
        actual   =   np.array(y)
        predicted =  self.predict(X)
    
        unique = sorted(set(actual))
        matrix = pd.DataFrame( [[0 for _ in unique] for _ in unique])
        imap   = {key: i for i, key in enumerate(unique)}
        # Generate Confusion Matrix
        for p, a in zip(predicted, actual):
            matrix[imap[p]][imap[a]] += 1

        return matrix

'''
np.random.seed(1204)
X = np.random.uniform(0,10,40).reshape(20,2)
y = np.random.choice(['a','b','c','d'],20)

knn_mod_3 = KNNClassifier(X,y,3)
print(knn_mod_3.predict(X))
print(knn_mod_3.score(X,y))


knn_mod_4 = KNNClassifier(X,y,4)
print(knn_mod_4.score(X,y))
print(knn_mod_4.predict(X))
'''

'''
np.random.seed(1548)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=2000, n_features=10, n_informative=4, n_clusters_per_class=1,class_sep=0.5,n_classes=6 )
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2)
X_val, X_test, y_val, y_test = train_test_split(X_holdout, y_holdout, test_size=0.5)

rng = range(1,100)
val_acc = []
for K in rng:
    temp_mod = KNNClassifier(X_train, y_train, K)
    val_acc.append(temp_mod.score(X_val,y_val))
    
plt.close()
plt.plot(rng, val_acc)
plt.show()
knn_mod = KNNClassifier(X_train, y_train, 10)
print("Training Accuracy:", knn_mod.score(X_train, y_train))
print("Testing Accuracy: ", knn_mod.score(X_test, y_test))
print(knn_mod.predict(X_test[:20,:]))
print(y_test[:20])
'''