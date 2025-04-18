import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, learning_rate: float, n_epochs: int, activation_function: function, init_weight_value=0.5):
        print("Initializing Perceptron")
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.activation_function = activation_function
        self.init_weight_value = init_weight_value
        print(f"Learning Rate: {self.learning_rate}")   
        print(f"Number of Epochs: {self.n_epochs}")
        


    def fit(self, X, y):
        # Loop over data n_epochs times
        # For each sample:
            # Calculate the weighted sum (z = x^T·w)
            # Apply activation function (e.g. step function)
            # Compute error (loss_function) (y_true - y_pred)
            # Update weights using Perceptron Learning Algorithm

        # Add bias neuron to input (always 1)
        X = self._add_bias(X)
        print(f"Input with Bias: {X}")

        # Initialize weights with its initial values - X.shape[1] is the number of features (input values)
        num_features = X.shape[1]
        self.weights = np.full(num_features, self.init_weight_value)
        print(f"Initial Weights: {self.weights}")

        # Loop over epochs
        for i in range(self.n_epochs):
            print('--'*20)
            print(f"Epoch {i+1}/{self.n_epochs}")
            num_weight_updates = 0 # Track how many weights were updated in this epoch, to check for convergence

            for j in range(X.shape[0]):
                # Get input and target output of the current instance
                X_i = X[j]
                y_i = y[j]
                print(f"Input: {X_i}, Output: {y_i}")
                
                # Calculate weighted sum
                z = self._calculate_weighted_sum(X_i)
                print(f"Weighted Sum: {z}")

                # Apply activation function
                y_pred = self._activation(z)
                print(f"Predicted Output: {y_pred}")

                # Calculate error score
                error = y_i - y_pred
                print(f"Error: {error}")

                # Update weights
                if error != 0:
                    # For each weight, update it using the Perceptron Learning Algorithm
                    # w_ij (next step) = w_ij + learning_rate * (y_i - y_pred) * x_j
                    for k in range(len(self.weights)):
                        self.weights[k] += self.learning_rate * error * X_i[k]
                    print(f"Updated Weights: {self.weights}")
                    num_weight_updates += 1
                else:
                    print("No weight update needed")
                
            if num_weight_updates == 0:
                # A whole epoch without weight updates means it has converged toward a solution
                print(f"Converged! After {i+1} epochs.")
                break
            else:
                if i == self.n_epochs - 1:
                    print(f"Maximum epochs reached WITHOUT any final solution. Weights updated {num_weight_updates} times.")
                    print(f"Final Weights: {self.weights}")
                else:
                    print(f"Number of Weight Updates: {num_weight_updates}")


        return self
    
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Use the trained model to make predictions on new data. 
        For each input, compute the weighted sum and apply the activation function to get the predicted class.
        Parameters
        ----------
            X : np.ndarray 
                Input data for prediction.
        
        Returns
        -------
        np.ndarray
            Predicted binary class (0 or 1) for each input.
        """

        num_predictions = X.shape[0]
        predictions = np.zeros(num_predictions, dtype=int)

        for i in range(num_predictions):
            X_i = X[i]

            # Calculate weighted sum
            z = self._calculate_weighted_sum(X_i)
            print(f"Weighted Sum: {z}")

            # Apply activation function
            y_pred = self._activation(z)
            print(f"Predicted Output: {y_pred}")

            # Store prediction
            predictions[i] = y_pred
            print(f"Updated Predictions: {predictions}")

        return predictions


    #region Utility methods    
    def _calculate_weighted_sum(self, X):
        """ Calculate the weighted sum of inputs and weights 
        Using formula: `z = x^T·w = x_1*w_1 + x_2*w_2 + ... + x_n*w_n`
        Where: X is a row vector (rækkevektor), and self.weights is a column vector (søjlevektor)
        """

        return np.dot(X, self.weights)

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """ Add (prepend) bias neuron to input (always 1) """
        bias = np.ones((X.shape[0], 1))
        X = np.hstack((bias, X))
        return X

    def _activation(self, weighted_sum):
        """ Apply activation function to a weighted sum """
        return self.activation_function(weighted_sum)
    #endregion

    #region Testing and Evaluation
    def score(self, y_pred, y_target):
        """ Calculate accuracy of the model on the given data """
        
         # TODO: Husk at bruge (tilføje) 'Statistik' i metode afsnit!
        accuracy = np.mean(y_pred == y_target)
        print(f"Accuracy: {accuracy}")

        return accuracy
    
    def confusion_matrix(self, y_true, y_pred):
        """ Calculate confusion matrix for binary classification """

        # True Positives, False Positives, True Negatives, False Negatives
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        confusion_matrix = np.array([[TP, FP],
                                      [FN, TN]])
        print(f"Confusion Matrix:\n{confusion_matrix}")
        
        return confusion_matrix
    
    def plot():
        # Plot to visualize decision boundaries (tærskelværdi) or training progress (loss vs epochs)
        pass

    def print_structure(self) -> None:
        """ Print the structure of the Perceptron (FOR DEBUGGING PURPOSES) """
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Number of Epochs: {self.n_epochs}")
        print(f"Activation Function: {self.activation_function}")
        print(f"Initial Weight Value: {self.init_weight_value}")
        print(f"Weights: {self.weights}")

    def get_weights(self) -> np.ndarray:
        """ Return the weights of the Perceptron """
        return self.weights

        

