import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init() # Initialize nnfs (sets random seed, etc.)

class Layer_Dense: # Fully connected (dense) layer
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # Initialize weights with small random values
        self.biases = np.zeros((1, n_neurons)) # Initialize biases with zeros

    def forward(self, inputs):
        self.inputs = inputs # Store input values
        self.output = np.dot(inputs, self.weights) + self.biases # Calculate output (weighted sum + bias)

    def backward(self, dvalues): 
        self.dweights = np.dot(self.inputs.T, dvalues) # Gradient of weights
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True) # Gradient of biases
        self.dinputs = np.dot(dvalues, self.weights.T) # Gradient of inputs

class Activation_ReLU: # ReLU activation function
    def forward(self, inputs):
        self.output = np.maximum(0, inputs) # Apply ReLU (set negative values to 0)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy() # Copy the gradient
        self.dinputs[self.inputs <= 0] = 0 # Zero gradient where input was <= 0

class Activation_Softmax: # Softmax activation for output layer
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # Exponentiate inputs (stabilized)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) # Normalize to probabilities
        self.output = probabilities # Store output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy() # Copy the gradient (not a true softmax backward)

class Loss: # Base loss class
    def calculate(self, output, y):
        sample_losses = self.forward(output, y) # Calculate loss for each sample
        data_loss = np.mean(sample_losses) # Average loss
        return data_loss # Return mean loss

class Loss_CategoricalCrossentropy(Loss): # Categorical cross-entropy loss
    def forward(self, y_pred, y_true):
        samples = y_pred.shape[0] # Number of samples
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) # Clip predictions to avoid log(0)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true] # Probabilities for correct class (sparse labels)
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1) # Probabilities for correct class (one-hot)
        negative_log_likelihoods = -np.log(correct_confidences) # Calculate negative log likelihood
        return negative_log_likelihoods # Return loss for each sample

    def backward(self, dvalues, y_true):
        samples = dvalues.shape[0] # Number of samples
        labels = dvalues.shape[1] # Number of labels/classes
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true] # Convert sparse to one-hot
        self.dinputs = -y_true / dvalues # Gradient of loss w.r.t. predictions
        self.dinputs = self.dinputs / samples    # Normalize gradients

X, y = spiral_data(samples=100, classes=3) # Generate spiral dataset (features X, labels y)

dense1 = Layer_Dense(2, 64) # First dense layer: 2 inputs (features), 64 neurons
activation1 = Activation_ReLU() # ReLU activation for first layer
dense2 = Layer_Dense(64, 64) # Second dense layer: 64 inputs, 64 neurons
activation2 = Activation_ReLU() # ReLU activation for second layer
dense3 = Layer_Dense(64, 3) # Output layer: 64 inputs, 3 neurons (classes)
activation3 = Activation_Softmax() # Softmax activation for output layer

dense1.forward(X) # Pass data through first dense layer
activation1.forward(dense1.output) # Apply ReLU activation
dense2.forward(activation1.output) # Pass through second dense layer
activation2.forward(dense2.output) # Apply ReLU activation
dense3.forward(activation2.output) # Pass through output layer
activation3.forward(dense3.output) # Apply softmax to get probabilities

print(activation3.output) # Print output probabilities

loss_function = Loss_CategoricalCrossentropy() # Create loss function instance
loss = loss_function.calculate(activation3.output, y) # Calculate loss
print(f'Loss: {loss}') # Print loss

# Visualize the spiral data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg') # Plot data points colored by class
plt.title('Spiral Data') # Set plot title
plt.xlabel('Feature 1') # X-axis label
plt.ylabel('Feature 2') # Y-axis label
plt.show() # Show plot
