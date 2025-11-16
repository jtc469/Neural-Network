import numpy as np
import tensorflow as tf
from network import Network, Layer
from activations import ReLU, Sigmoid, Linear


def predict_x2_scratch(net, x_raw):
    x_scaled = (x_raw - 5.0) / 5.0
    y_pred = net.forward(np.array([x_scaled]))[0]
    return y_pred


def predict_x2_tf(model, x_raw):
    x_scaled = (x_raw - 5.0) / 5.0
    x_input = np.array([[x_scaled]], dtype=np.float32)
    y_pred = model.predict(x_input, verbose=0)[0, 0]
    return y_pred


def test():
    net = Network([
        Layer(input_dim=1, output_dim=16, activation=ReLU()),
        Layer(input_dim=16, output_dim=16, activation=ReLU()),
        Layer(input_dim=16, output_dim=1, activation=Linear())
    ])

    net.load("models/model.npz")
    tf_model = tf.keras.models.load_model("models/tf_model.keras")

    test_vals = np.arange(1, 11)

    for t in test_vals:
        p_scratch = predict_x2_scratch(net, t)
        p_tf = predict_x2_tf(tf_model, t)
        y_true = t**2
        print(f"{t} -> scratch: {p_scratch:.4f}, tf: {p_tf:.4f}, true: {y_true:.4f}")
