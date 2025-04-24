import numpy as np
import pickle


class PyPerceptron:
    def __init__(self, learning_rate: float, n_epochs: int, activation_function: object, init_weight_value=0.5, verbose=True, save_path='perceptron_model.pkl'):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.activation_function = activation_function
        self.init_weight_value = init_weight_value
        self.verbose = verbose
        self.save_path = save_path
        self.weights = None # Will be initialized when fitting

 
    def fit(self, X) -> np.ndarray: 
        """ Fit the Perceptron model to the training data. 
        Parameters
        ----------
            X : np.ndarray 
                Input data for training.
        Returns
        -------
        np.ndarray
            Input data with bias neuron added.
        """

        # Add bias neuron to input (always 1)
        X = self._add_bias(X)

        # Initialize weights with its initial values - X.shape[1] is the number of features (input values)
        num_features = X.shape[1]
        self.weights = np.full(num_features, self.init_weight_value) # Fills the array with the initial weight value

        return X
    

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Train the Perceptron model on the provided data.
        Parameters
        ----------
            X : np.ndarray 
                Input data for training.
            y : np.ndarray 
                Target output for training.
        """

        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must be equal.")
        if X.shape[1] != self.weights.shape[0]:
            raise ValueError("Number of features in X must match the number of weights. Try fitting the model first.")
        if len(y.shape) != 1:
            raise ValueError("y must be a 1D array.")
        
        # Loop over epochs
        for i in range(self.n_epochs):
            self._print('--'*20)
            self._print(f"Epoch {i+1}/{self.n_epochs}")
            num_weight_updates = 0 # Track how many weights were updated in this epoch, to check for convergence

            for j in range(X.shape[0]):
                # Get input and target output of the current instance
                X_i = X[j]
                y_i = y[j]
                self._print(f"Input: {X_i}, Output: {y_i}")
                
                # Calculate weighted sum
                z = self._calculate_weighted_sum(X_i)
                self._print(f"Weighted Sum: {z}")

                # Apply activation function
                y_pred = self._activation(z)
                self._print(f"Predicted Output: {y_pred}")

                # Calculate error score
                error = y_i - y_pred
                self._print(f"Error: {error}")

                # Update weights
                if error != 0:
                    # For each weight, update it using the Perceptron Learning Algorithm
                    # w_ij (next step) = w_ij + learning_rate * (y_i - y_pred) * x_j
                    for k in range(len(self.weights)):
                        self.weights[k] += self.learning_rate * error * X_i[k]
                    self._print(f"Updated Weights: {self.weights}")
                    num_weight_updates += 1
                else:
                    self._print("No weight update needed")
                
            if num_weight_updates == 0:
                # A whole epoch without weight updates means it has converged toward a solution
                self._print(f"Converged! After {i+1} epochs.")
                self._print(f"Final Weights: {self.weights}")
                # Save with pickle
                pickle.dump(self, open(self.save_path, 'wb'))
                print(f"Model saved to {self.save_path}")
                break
            else:
                if i == self.n_epochs - 1:
                    print(f'Iteration limit reached. No convergence. {i} epochs out of {self.n_epochs} epochs.')
                    print(f"Maximum epochs reached WITHOUT any final solution. Weights updated {num_weight_updates} times.")
                    print(f"Final Weights: {self.weights}")
                else:
                    self._print(f"Number of Weight Updates: {num_weight_updates}")
        
    
    
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

        print(X.shape)
        print(self.weights.shape)
        print('---' * 20)

        print(X)
        print(self.weights)
        print('---' * 20)

        # If bias neuron is not added, add it
        if X.shape[1] != self.weights.shape[0]:
            X = self._add_bias(X)
            self._print(f"Bias neuron added to input data: {X}")

        num_predictions = X.shape[0]
        predictions = np.zeros(num_predictions, dtype=int)

        for i in range(num_predictions):
            X_i = X[i]

            # Calculate weighted sum
            z = self._calculate_weighted_sum(X_i)
            self._print(f"Weighted Sum: {z}")

            # Apply activation function
            y_pred = self._activation(z)
            self._print(f"Predicted Output: {y_pred}")

            # Store prediction
            predictions[i] = y_pred
            self._print(f"Updated Predictions: {predictions}")

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
        accuracy = np.mean(y_pred == y_target)
        self._print(f"Accuracy: {accuracy}")

        precision = np.sum((y_pred == 1) & (y_target == 1)) / np.sum(y_pred == 1)
        recall = np.sum((y_pred == 1) & (y_target == 1)) / np.sum(y_target == 1)
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")

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
        self._print(f"Confusion Matrix:\n{confusion_matrix}")
        
        return confusion_matrix
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """ Plot confusion matrix """
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        cm = self.confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap=plt.cm.Blues)
        plt.show()
    
    def plot():
        # Plot to visualize decision boundaries (tærskelværdi) or training progress (loss vs epochs)
        pass

    def print_structure(self) -> None:
        """ Print the structure of the Perceptron (FOR DEBUGGING PURPOSES) """
        self._print(f"Learning Rate: {self.learning_rate}")
        self._print(f"Number of Epochs: {self.n_epochs}")
        self._print(f"Activation Function: {self.activation_function}")
        self._print(f"Initial Weight Value: {self.init_weight_value}")
        self._print(f"Weights: {self.weights}")

    def get_weights(self) -> np.ndarray:
        """ Return the weights of the Perceptron """
        return self.weights

    def _print(self, message: str) -> None:
        """ Print a message if verbose is enabled """
        if self.verbose:
            print(message)
        

