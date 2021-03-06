#Aaron Henningfeld
#Program 2

import numpy as np 
import pandas as pd
from scipy.optimize import minimize

class LogisticRegression:

    def __init__(self, X, y):
        
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_observations = len(X)
        self.classes = np.unique(self.y)
        self.classZero = self.classes[0]
        self.classOne = self.classes[1]
        
        def find_neg_loglik(coeff):
            
            beta = np.array(coeff)
            
            z = beta[0] + np.sum(beta[1:] * self.X, axis=1)
            p = 1 / (1 + np.exp(-z)) #gives the probability that each observation lands in class 1
            pi = np.where(self.y == self.classOne, p, 1 - p) 
            pi = np.where(pi == 0, 1e-100, pi) #used to make sure there are no zeros in the array
            self.loglik = np.sum(np.log(pi))
            return -self.loglik
        
        np.seterr(all='ignore')
        beta_guess = np.zeros(X.shape[1] + 1) 
        min_results = minimize(find_neg_loglik, beta_guess)   
        self.coefficients = min_results.x
        np.seterr(all='warn')
        
        self.accuracy = self.score(self.X, self.y)        

        
    def predict_proba(self, X):
        
        X = np.array(X)
        z = self.coefficients[0] + np.sum(self.coefficients[1:] * X, axis=1)
        p = 1 / (1 + np.exp(-z)) #gives the probability that each observation lands in class 1
        
        return p
    
    def predict(self, X, t=0.5):
        prob = self.predict_proba(X)
        y_pred = np.where(prob>t, self.classes[1], self.classes[0])
        
        return y_pred
    
    def score(self, X, y, t=0.5):
        y=np.array(y)
        y_pred = self.predict(X,t)
        numEqual = np.sum(y == y_pred)
        accuracy = numEqual/len(y)
        return accuracy
    
    def summary(self):
        print('+------------------------------+')
        print('| Logistic Regression Summary  |')
        print('+------------------------------+')
        print('Number of training observations: ', self.n_observations)
        print('Coefficient Estimates: ', self.coefficients)
        print('Log-Likelihood:', self.loglik)
        print('Accuracy:', self.accuracy, '\n')
        return
    
    def confusion_matrix(self, X, y, t=0.5):
        actual = np.array(y)
        predicted = self.predict(X,t)

        TP = np.sum((actual==self.classOne)&(predicted==self.classOne))
        FP = np.sum((actual==self.classZero)&(predicted==self.classOne))

        TN = np.sum((actual==self.classZero)&(predicted==self.classZero))
        FN = np.sum((actual==self.classOne)&(predicted==self.classZero))

        cm = pd.DataFrame([[TN,FP],[FN,TP]])
        cm.columns = ['Pred_0', 'Pred_1'] 
        cm.index = ['True_0', 'True_1']
        print('Class 0:',self.classZero)
        print('Class 1:', self.classOne)
        print(cm)
        return
