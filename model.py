import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(x.dtype)


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


class AffineLayer:
    def __init__(self, input_size, output_size, activation=None):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        self.activation_deriv = None

    def forward(self, x):
        self.x = x  # Cache the input for use in the backward pass
        z = np.dot(x, self.weights) + self.biases
        self.z = z  # Cache the pre-activation output for backprop
        if self.activation == 'relu':
            return relu(z)
        elif self.activation == 'softmax':
            return softmax(z)
        return z

    def backward(self, grad_output, learning_rate):
        if self.activation == 'relu':
            grad_output *= relu_derivative(self.z)

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


class MultiLayerNetwork:
    def __init__(self, layer_sizes, initial_learning_rate=0.05, decay_rate=0.9, decay_step=50):
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.decay_step = decay_step
        self.layers = []
        for i in range(len(layer_sizes) - 2):
            self.layers.append(AffineLayer(layer_sizes[i], layer_sizes[i + 1], activation='relu'))
        # Softmax activation on the last layer
        self.layers.append(AffineLayer(layer_sizes[-2], layer_sizes[-1], activation='softmax'))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def compute_loss_and_gradients(self, logits, y_true):
        m = y_true.shape[0]  # Number of examples

        # Shift logits for numerical stability
        logits -= np.max(logits, axis=1, keepdims=True)

        # Compute exponential of the shifted logits
        exp_logits = np.exp(logits)

        # Avoid division by zero in softmax calculation
        sum_exp_logits = np.sum(exp_logits, axis=1, keepdims=True)
        sum_exp_logits = np.clip(sum_exp_logits, 1e-12, np.inf)  # Clip to prevent division by zero

        # Compute probabilities
        probabilities = exp_logits / sum_exp_logits

        # Compute cross-entropy loss
        # Using clipping to prevent log(0) scenario
        probabilities_clipped = np.clip(probabilities, 1e-12, 1.0)
        log_likelihood = -np.log(probabilities_clipped[range(m), y_true.argmax(axis=1)])
        loss = np.sum(log_likelihood) / m

        # Gradient of loss w.r.t. softmax input
        grad_softmax = probabilities
        grad_softmax[range(m), y_true.argmax(axis=1)] -= 1
        grad_softmax /= m

        return loss, grad_softmax

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, self.learning_rate)

    def train(self, X_train, Y_train, epochs):
        for epoch in range(epochs):
            if epoch % self.decay_step == 0 and epoch > 0:
                self.learning_rate *= self.decay_rate
                print(f"Learning rate decayed to: {self.learning_rate}")

            logits = self.forward(X_train)
            loss, grad_softmax = self.compute_loss_and_gradients(logits, Y_train)
            self.backward(grad_softmax)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
