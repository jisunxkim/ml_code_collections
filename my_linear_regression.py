from sklearn import metrics
from matplotlib import pyplot as plt
from my_get_data import get_data
import numpy as np
from collections import deque

class SimpleLinearRegression:
    def __init__(self):
        self.parameters ={
            'slope': np.random.random(),
            'constant': np.random.random()
        }
        self.colors = deque(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'])
        self.marker = deque(['.', 'o', 'v', '^', '<', '>', '*', 'h', 'd', 'P', 's'])
    def forward_propagation(self, train_input):
        """
        Prediction with the current parameters
        """
        slope = self.parameters['slope']
        constant = self.parameters['constant']        
        predictions = np.multiply(slope, train_input) + constant
        return predictions
    
    def cost_function(self, predictions, train_ouput):
        """
        Mean Square Error loss function
        """
        mse = np.mean((predictions - train_ouput) ** 2)
        return mse
    
    def backward_propagation(self, train_input, train_output, predictions):
        """
        Calculate gradients given the current parameters and predictions
        return a dictionary of gradients of the parameters
        gradient:
            derivative of constant = sum((predictions-train_output)) 
            derivative of slope = sum((predictions-train_output)*train_input)
        """
        derivatives = {}
        derivatives['constant'] = np.mean(predictions - train_output)
        derivatives['slope'] = np.mean(np.multiply((predictions - train_output), train_input))
        return derivatives
   
    def update_parameters(self, derivatives, learning_rate):
        """
        Update model parameters using gradients dervaties and learning rate.
        """ 
        self.parameters['constant'] = self.parameters['constant'] - \
            learning_rate * derivatives['constant'] 
        self.parameters['slope'] = self.parameters['slope'] - \
            learning_rate * derivatives['slope']
     
    
    def train(self, train_input, train_ouput, n_epoch, learning_rate):
        for epoch in range(n_epoch+1):
            predictions = self.forward_propagation(train_input)
            cost = self.cost_function(predictions, train_ouput)
            print(f'epoch = {epoch}', '_'*10)
            self.evaluation(train_ouput, predictions, epoch)
            print(f'Loss: {cost:0.4f}')

            if epoch < n_epoch:
                print(f'epoch {epoch+1} updating parameters', '.'*10)
                derivatives = self.backward_propagation(train_input, train_ouput, predictions)
                self.update_parameters(derivatives, learning_rate)
        
        plt.show()
        
    def evaluation(self, y_data, predictions, n_iter):
        r2_score = metrics.r2_score(y_data, predictions)
        print(f'R2 Score: {r2_score:0.4f}', end=' | ')
        
        cur_marker = self.marker.popleft()
        self.marker.append(cur_marker)
        cur_color = self.colors.popleft()
        self.colors.append(cur_color)
        plt.scatter(y_data, predictions, s=0.2, marker=cur_marker,c=cur_color)
        plt.plot(y_data, y_data, c='r', label='Perfect Model', alpha=0.2)
        plt.xlabel('actual')
        plt.ylabel('prediction')
        
    
if __name__ == '__main__':
    data = get_data().dropna()
    simple_lr = SimpleLinearRegression()
    simple_lr.train(data['x'], data['y'], n_epoch = 10, learning_rate=0.0001)
    
