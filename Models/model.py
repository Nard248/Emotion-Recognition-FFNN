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

    def forward(self, x):
        self.x = x
        z = np.dot(x, self.weights) + self.biases
        self.z = z
        if self.activation == 'relu':
            self.output = relu(z)
        elif self.activation == 'softmax':
            self.output = softmax(z)
        else:
            self.output = z
        return self.output

    def backward(self, grad_output, learning_rate, max_norm=1.0):
        if self.activation == 'relu':
            grad_output *= relu_derivative(self.z)

        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(self.x.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0, keepdims=True)

        # Gradient clipping
        grad_weights = np.clip(grad_weights, -max_norm, max_norm)
        grad_biases = np.clip(grad_biases, -max_norm, max_norm)

        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input

class MultiLayerNetwork:
    def __init__(self, layer_sizes, initial_learning_rate=0.01, decay_rate=0.9, decay_step=50):
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.decay_step = decay_step
        self.layers = [AffineLayer(layer_sizes[i], layer_sizes[i + 1], activation='relu' if i < len(layer_sizes) - 2 else 'softmax')
                       for i in range(len(layer_sizes) - 1)]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def compute_loss_and_gradients(self, outputs, y_true):
        logits = outputs[-1]  # Output from the last softmax layer
        m = y_true.shape[0]  # Number of examples
        probabilities = logits
        probabilities_clipped = np.clip(probabilities, 1e-12, 1.0)

        log_likelihood = -np.log(probabilities_clipped[range(m), y_true.argmax(axis=1)])
        loss = np.sum(log_likelihood) / m

        # Gradient of loss w.r.t. softmax input
        grad_softmax = probabilities.copy()
        grad_softmax[range(m), y_true.argmax(axis=1)] -= 1
        grad_softmax /= m

        return loss, grad_softmax

    def backward(self, grad_output, outputs):
        for layer, output in zip(reversed(self.layers), reversed(outputs[:-1])):
            grad_output = layer.backward(grad_output, self.learning_rate)

    def train(self, X_train, Y_train, epochs):
        for epoch in range(epochs):
            if epoch % self.decay_step == 0 and epoch > 0:
                self.learning_rate *= self.decay_rate
                print(f"Learning rate decayed to: {self.learning_rate}")

            outputs = self.forward(X_train)
            loss, grad_softmax = self.compute_loss_and_gradients(outputs, Y_train)
            self.backward(grad_softmax, outputs)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")