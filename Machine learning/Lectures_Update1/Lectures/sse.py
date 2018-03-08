import numpy as np 

# We start by definining X, y, and beta.
X = np.array([[12,4], [14,3], [16,6], [20,5], [24,2]])
y = np.array([50, 53, 67, 70, 63])
beta = np.array([12, 1.5, 5])

def find_sse(coeff):
            beta = np.array(coeff)
            
            # We start by definining X, y, and beta.
            X2 = X
            y2 = y

            # Now we calculate y_hat and sse.
            y_hat = beta[0] + np.sum(beta[1:] * X2, axis=1)
            residuals = y2 - y_hat
            sse = np.sum(residuals**2)
            
            # Print the results
            print('y_hat = ', y_hat)
            print('residuals = ', residuals)
            print('sse = ', sse)
            
            return sse
        
beta_guess = np.zeros(X.shape[1] + 1) 
        
print(beta_guess)
        
find_sse(beta)
        
        #min_results = minimize(find_sse(beta_guess)) 
        #self.coefficients = min_results.x