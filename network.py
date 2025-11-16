import numpy as np

class Layer:
    def __init__(self, input_dim, output_dim, activation):
        
        # Initialise random weight and bias vectors
        self.W = np.random.randn(output_dim, input_dim) * 0.01
        self.b = np.zeros(output_dim)
        self.activation = activation

        # Input and output vectors
        self.x = None
        self.z = None

    def forward(self, x):
        self.x = x
        self.z = np.dot(self.W, x) + self.b
        return self.activation(self.z)

    def backward(self, grad_output, lr):
        """
        Backpropogation logic from "Neural Networks.pdf"
        
        Chain rule to calculate dz as a multiple of W, activation, and W^[l-1]
        """
        dz = grad_output * self.activation.d(self.z)
        grad_W = np.outer(dz, self.x) # Using outer product https://www.geeksforgeeks.org/python/outer-product-on-vector/
        grad_b = dz
        grad_input = np.dot(self.W.T, dz)
        self.W -= lr * grad_W
        self.b -= lr * grad_b
        return grad_input
    

class Network:
    """
    Networks are set of layers
    - move forward and backwards by moving through each layer in the network.


    Save and load so we don't have to retrain the network for each test
    Network properties are saved in a hashmap format (.npz)
    https://numpy.org/doc/2.2/reference/generated/numpy.savez.html
    """

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output, lr):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, lr)
    
    def save(self, path):
        data = {}
        for i, layer in enumerate(self.layers):
            data[f"W{i}"] = layer.W
            data[f"b{i}"] = layer.b
        np.savez(path, **data)

    def load(self, path):
        saved = np.load(path)
        for i, layer in enumerate(self.layers):
            layer.W = saved[f"W{i}"]
            layer.b = saved[f"b{i}"]
