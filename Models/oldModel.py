import numpy as np


class AffineLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, x):
        self.x = x  # Cache the input for use in the backward pass
        return np.dot(x, self.weights) + self.biases

    def backward(self, grad_output, learning_rate):
        # Gradient with respect to input
        grad_input = np.dot(grad_output, self.weights.T)
        # Gradient with respect to weights
        grad_weights = np.dot(self.x.T, grad_output)
        # Gradient with respect to biases
        grad_biases = np.sum(grad_output, axis=0, keepdims=True)
        # Update parameters
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases
        return grad_input


def compute_loss_and_gradients(logits, y_true):
    # Stable computation of softmax followed by cross-entropy loss
    m = y_true.shape[0]
    # Shift logits for numerical stability
    logits -= np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    # Cross-entropy loss
    log_likelihood = -np.log(probabilities[range(m), y_true.argmax(axis=1)] + 1e-9)
    loss = np.sum(log_likelihood) / m
    # Gradient of loss w.r.t. softmax input
    grad_softmax = probabilities
    grad_softmax[range(m), y_true.argmax(axis=1)] -= 1
    grad_softmax /= m
    return loss, grad_softmax


class MultiLayerNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(AffineLayer(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, self.learning_rate)

    def train(self, X_train, Y_train, epochs):
        for epoch in range(epochs):
            logits = self.forward(X_train)
            loss, grad_softmax = compute_loss_and_gradients(logits, Y_train)
            self.backward(grad_softmax)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")


# checking on samole data
def create_data(num_samples, num_classes):
    # Generate synthetic data for training.
    np.random.seed(0)  # Seed for reproducibility.
    X = np.random.randn(num_samples, 2)  # Random 2D points.
    Y = np.zeros((num_samples, num_classes))
    # Assign labels based on the sum of absolute values of the coordinates.
    distances = np.sum(np.abs(X), axis=1)
    labels = (distances / max(distances) * (num_classes - 1)).astype(int)
    for i, label in enumerate(labels):
        Y[i, label] = 1
    return X, Y
#
# # Create synthetic data.
# num_samples = 1000
# num_classes = 3
# X_train, Y_train = create_data(num_samples, num_classes)
#
# # Initialize the neural network.
# layer_sizes = [2, 50, 30, num_classes]  # -1 indicates the input size is to be determined by the data.
# nn = MultiLayerNetwork(layer_sizes, learning_rate=0.1)
#
# # Train the network.
# nn.train(X_train, Y_train, epochs=200)
#
# # Function to evaluate the network's performance.
# def evaluate(X, Y):
#     predictions = nn.forward(X)
#     predicted_classes = np.argmax(predictions, axis=1)
#     true_classes = np.argmax(Y, axis=1)
#     accuracy = np.mean(predicted_classes == true_classes)
#     return accuracy
#
# # Evaluate and print training accuracy.
# accuracy = evaluate(X_train, Y_train)
# print(f"Training Accuracy: {accuracy:.2f}")
#
#
# This is the model I have written, do you understand the code?