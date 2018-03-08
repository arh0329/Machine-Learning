import numpy as np
import pandas as pd
from scipy.optimize import minimize

class LogisticRegression:
    ''' This class implements the binary logistic regression algorithm.'''
    
    def __init__(self, X, y):
        ''' Constructor '''
    
        # Objective function used to score models.
        def find_neg_loglik(beta):
            z = beta[0] + np.sum(beta[1:] * self.X, axis=1)      
            p = 1 / (1 + np.exp(-z))
            pi = np.where(self.y == self.classes[1], p, 1 - p)
            loglik = np.sum(np.log(pi))
            return -loglik
        
        # Store training data.
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_observations = len(self.y)
        
        # Determine possible classes.
        self.classes = np.unique(self.y)
        
        # Find the optimal coefficient values. 
        np.seterr(all='ignore')
        beta_guess = np.zeros(self.X.shape[1] + 1)
        min_results = minimize(find_neg_loglik, beta_guess)
        self.coefficients = min_results.x
        np.seterr(all='warn')
        
        # Generate predictions
        self.y_predicted = self.predict(self.X)

        # Calculate performance metrics
        self.accuracy = self.score(self.X, self.y)
        self.loglik = -find_neg_loglik(self.coefficients)

    def predict_proba(self, X):
        ''' Predicts the probability of instances in X belonging to Class 1.'''
        X = np.array(X)
        z = self.coefficients[0] + np.sum(self.coefficients[1:]*X, axis=1)
        prob = 1 / (1 + np.exp(-z))
        return prob
               
    def predict(self, X, t=0.5):
        ''' Returns the predicted classes of instances in X.'''
        prob = self.predict_proba(X)
        y_pred = np.where(prob >= t, self.classes[1], self.classes[0])
        return y_pred
        
    
    def summary(self):
        ''' Provides a summary of the trained model. '''
        print('+-------------------------------+')
        print('|  Logistic Regression Summary  |')
        print('+-------------------------------+')
        print('Number of training observations:', self.n_observations)
        print('Coefficient Estimates:\n  ', self.coefficients)
        print('Log-Likelihood:', self.loglik)
        print('Accuracy:', self.accuracy)
        print()
    
    
    def score(self, X, y, t=0.5):
        ''' Returns the model accuracy, calculated on the data provided. '''
        X = np.array(X)
        y = np.array(y)        
        y_predicted = self.predict(X,t)  
        return np.sum(y == y_predicted) / len(y)   
    

    def confusion_matrix(self, X, y, t=0.5):
        ''' Prints a confustion matrix based on the supplied data.'''
        X = np.array(X)
        y = np.array(y)
        
        y_pred = self.predict(X, t)
               
        tp = sum((y_pred == self.classes[1]) & (y == self.classes[1])) 
        fp = sum((y_pred == self.classes[1]) & (y == self.classes[0]))
        tn = sum((y_pred == self.classes[0]) & (y == self.classes[0]))
        fn = sum((y_pred == self.classes[0]) & (y == self.classes[1]))

        cm = pd.DataFrame([[tn, fp],[fn, tp]])
        cm.columns = ['Pred_0', 'Pred_1']
        cm.index = ['True_0', 'True_1']

        print('Class 0:', self.classes[0])        
        print('Class 1:', self.classes[1])        
        print('\n', cm, '\n')
        
        
