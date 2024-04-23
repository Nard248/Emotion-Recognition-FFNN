import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(x.dtype)


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


class Conv1DLayer:
    def __init__(self, num_filters, filter_size, stride=1):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        # Initialize the filters with a proper scaling factor
        self.filters = np.random.randn(num_filters, filter_size) * np.sqrt(2. / (num_filters * filter_size))

    def forward(self, x):
        self.x = x
        num_samples, feature_length, channels = x.shape
        output_length = (feature_length - self.filter_size) // self.stride + 1
        self.output = np.zeros((num_samples, output_length, self.num_filters))

        for i in range(output_length):
            # Perform the convolution operation for each segment of the input
            # Create a 3D array to store the results for each filter
            conv_results = np.zeros((num_samples, 1, self.num_filters))

            for f in range(self.num_filters):
                # Convolve the filter over the sequence and store the results
                conv_results[:, 0, f] = np.sum(
                    x[:, (i * self.stride):(i * self.stride + self.filter_size)] *
                    self.filters[f, :],
                    axis=1
                )

            # Apply the ReLU activation function to the convolution results
            self.output[:, i, :] = relu(conv_results[:, 0, :])

        return self.output

    def backward(self, d_out, learning_rate):
        d_filters = np.zeros(self.filters.shape)
        d_out = d_out * relu_derivative(self.output)

        output_length = d_out.shape[1]

        for i in range(output_length):
            d_filters += np.tensordot(
                self.x[:, (i * self.stride):(i * self.stride + self.filter_size), :],
                d_out[:, i, :],
                axes=[0, 0]
            )

        self.filters -= learning_rate * d_filters

        # Backprop to inputs not implemented for simplicity
        return None  # This would be the gradient with respect to the input, for chaining layers.


class FlattenLayer:
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, d_out, learning_rate):
        return d_out.reshape(self.input_shape)


class FullyConnectedLayer:
    def __init__(self, input_size, output_size, activation=None):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((1, output_size))
        self.activation = activation

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

        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(self.x.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0, keepdims=True)

        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input
class MultiLayerConvNet:
    def __init__(self, layer_configs, input_dim, num_classes, initial_learning_rate=0.01, decay_rate=0.9,
                 decay_step=50):
        self.learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.decay_step = decay_step
        self.layers = []

        # Create layers based on the provided configuration
        for config in layer_configs:
            if config['type'] == 'conv1d':
                layer = Conv1DLayer(config['num_filters'], config['filter_size'], config['stride'])
            elif config['type'] == 'flatten':
                layer = FlattenLayer()
            elif config['type'] == 'affine':
                layer = FullyConnectedLayer(config['input_size'], config['output_size'], config['activation'])
            self.layers.append(layer)

        # Final softmax layer
        self.layers.append(FullyConnectedLayer(layer_configs[-1]['output_size'], num_classes, 'softmax'))

    def forward(self, x):
        # Forward pass through all layers
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def compute_loss(self, predictions, y_true):
        # Cross-entropy loss calculation
        loss = -np.sum(y_true * np.log(predictions + 1e-15)) / predictions.shape[0]
        return loss

    def backward(self, y_true):
        # Backward pass through all layers
        grad_y = y_true
        for layer in reversed(self.layers):
            grad_y = layer.backward(grad_y, self.learning_rate)

    def train(self, X_train, Y_train, epochs):
        for epoch in range(epochs):
            predictions = self.forward(X_train)
            loss = self.compute_loss(predictions, Y_train)
            self.backward(Y_train)
            if epoch % self.decay_step == 0 and epoch > 0:
                self.learning_rate *= self.decay_rate
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')


