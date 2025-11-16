import numpy as np
from network import Network, Layer
from activations import ReLU, Sigmoid, Linear



def test():
    net = Network([
    Layer(input_dim=1, output_dim=16, activation=ReLU()),
    Layer(input_dim=16, output_dim=16, activation=ReLU()),
    Layer(input_dim=16, output_dim=1, activation=Linear())
])

    net.load("models/model.npz")
    def predict_x2(net, x_raw):
        x_scaled = (x_raw - 5.0) / 5.0
        y_scaled = net.forward(np.array([x_scaled]))[0]
        return y_scaled

    test = np.array(range(10))
    for t in test:
        p = predict_x2(net, t)
        print(f"{t} -> {p:.4f}")