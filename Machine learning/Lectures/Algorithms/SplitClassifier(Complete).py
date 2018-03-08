import numpy as np

class SplitClassifier:

    def __init__(self, X, y):
        
        

        self.data = np.array(X)
        self.labels = np.array(y)
        self.size = len(y)
 
        # Find and order possible label categories
        self.classes = sorted(set(self.labels))

        # Initialize the Training Accuracy to 0
        self.accuracy = 0
        
        # Iterate over each axis/feature
        for i in range(self.data.shape[1]):

            # Obtain sorted list of feature values                        
            col_values = self.data[:,i].copy()
            col_values.sort()
           
            # Iterate over each observation
            for j in range(self.size):
                
                # Select values below the current observation
                sel = self.data[:,i] <= col_values[j]
                #sel = self.data.iloc[:,i].values <= col_values[j]
                
                # Determine the number correctly classified, assuming
                # that the lower class is class[0]
                n_correct = (np.sum(self.labels[sel] == self.classes[0]) + 
                             np.sum(self.labels[~sel] == self.classes[1]))

                # Determine the accuracy for the current cut
                temp_acc = n_correct / self.size
                cur_acc = max(temp_acc, 1 - temp_acc)
                
                # If new cut is an improvement, update attributes
                if cur_acc >= self.accuracy:
                    self.accuracy = cur_acc
                    self.axis = i
                    
                    if(j == len(col_values) - 1):
                        self.threshold = col_values[j]
                    else:
                        self.threshold = 0.5*(col_values[j] + col_values[j+1])               
                    
                    if cur_acc == temp_acc:
                        self.lower_class = self.classes[0]
                        self.upper_class = self.classes[1]
                    else:
                        self.lower_class = self.classes[1]
                        self.upper_class = self.classes[0]
   

    def predict(self, X):

        # Create inner function to classify an individual observation
        def classify_obs(row):
            if row[self.axis] <= self.threshold:
                return self.lower_class
            else:
                return self.upper_class                

        # Convert X to a NumPy array
        X = np.array(X)
        
        # Apply classify_obs to rows of X
        return np.apply_along_axis(classify_obs, 1, X)
    
    
    def score(self, X, y):
        
        X = np.array(X)
        y = np.array(y)
        
        predictions = self.predict(X)
        num_correct = np.sum(predictions == y)
        
        return num_correct / len(y)
        
        
    def summary(self):
        print('+----------------------------+')
        print('|  Split Classifier Summary  |')
        print('+----------------------------+')
        print('Number of training observations:', self.size)
        print('Axis/Feature split:', self.axis)
        print('Threshold value:', self.threshold)
        print('Predicted class below threshold:', self.lower_class)
        print('Predicted class above threshold:', self.upper_class)
        print('Training accuracy:', self.accuracy, '\n')