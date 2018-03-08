import numpy as np 
from scipy.optimize import minimize 
from scipy.stats import norm
import math

class LinearRegression:

    def __init__(self, X, y):
        #always work with numpy arrays
        self.data = np.array(X) #get data from X
        self.labels = np.array(y) #get labels from y
        self.n_observations = len(X) #find # of observations
        
        def find_sse(coeff):
            beta = np.array(coeff) #array of coeff guesses
            # Now we calculate y_hat and sse.
            y_hat = beta[0] + np.sum(beta[1:] * self.data, axis=1)
            self.residuals = self.labels - y_hat #find the errors
            sse = np.sum(self.residuals**2) #find sse
            return sse
        
        beta_guess = np.zeros(X.shape[1] + 1) #create guess of all zeros
        min_results = minimize(find_sse, beta_guess) #pass into minimize
        self.sse = min_results.fun #find sse
        self.sst = np.sum((self.labels-np.mean(self.labels))**2) #find sst
        self.coefficients = min_results.x #find coeff
        self.rse=math.sqrt(self.sse/(self.n_observations-2)) #find rse
        self.r_squared = 1-(self.sse/self.sst) # find r^2
        self.loglik = np.sum(np.log(norm.pdf(self.residuals, 0, self.rse)))

    def predict(self, X):                 
        # Convert X to a NumPy array and find predicted Ys
        X = np.array(X)
        self.y_predicted = self.coefficients[0] + np.sum(self.coefficients[1:]*X, axis=1)
        return self.y_predicted
    
    def score(self, X, y):
        # Convert to NumPy array and find r^2
        X = np.array(X)
        y = np.array(y)
        y_predicted = self.predict(X) #predicted y
        errors = y-y_predicted #find error
        sse=np.sum(errors**2)
        sst = np.sum((y-np.mean(y))**2)
        r_squared = 1-(sse/sst)
        return r_squared
        
    def summary(self):
        print('+----------------------------+')
        print('| Linear Regression Summary  |')
        print('+----------------------------+')
        print('Number of training observations: ', self.n_observations)
        print('Coefficient Estimates: ', self.coefficients)
        print('Residual Standard Error: ', self.rse)
        print('r-Squared: ', self.r_squared)
        print('Log-Likelihood:', self.loglik, '\n')