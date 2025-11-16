import numpy as np
import tensorflow as tf
from network import Network, Layer
from activations import ReLU, Sigmoid, Linear
from losses import MSE


def train():
    relu = ReLU()
    sig = Sigmoid()
    linear = Linear()
    loss_fn = MSE()

    loss_tol = 1e-3
    plateau_tol = 1e-5
    repeat_loss = 500
    losses = []

    net = Network([
        Layer(input_dim=1, output_dim=16, activation=relu),
        Layer(input_dim=16, output_dim=16, activation=relu),
        Layer(input_dim=16, output_dim=1, activation=linear)
    ])

    xs_raw = np.linspace(0, 10, 500)
    noise = 0.1 * np.random.randn(xs_raw.size)
    ys_true = xs_raw**2 + noise
    xs = (xs_raw - 5.0) / 5.0

    lr = 5e-4
    epochs = 4000

    for epoch in range(epochs):
        loss = 0.0
        for x_val, y_val in zip(xs, ys_true):
            x_vec = np.array([x_val])
            y_vec = np.array([y_val])
            pred = net.forward(x_vec)
            loss += loss_fn(pred, y_vec)
            grad = loss_fn.d(pred, y_vec)
            net.backward(grad, lr)
        loss /= len(xs)
        losses.append(loss)

        if epoch % 100 == 0:
            print(f"* SCRATCH TRAINING: {epoch}/{epochs} (LOSS: {loss:.6f})")

        if loss < loss_tol:
            print(f"SCRATCH EARLY STOP (tolerance) at epoch {epoch}, loss {loss:.6f}")
            break

        if len(losses) >= repeat_loss:
            recent = losses[-repeat_loss:]
            if max(recent) - min(recent) < plateau_tol:
                print(
                    f"SCRATCH EARLY STOP (plateau) at epoch {epoch}, loss {loss:.6f} \n"
                    f"Loss range {max(recent) - min(recent):.6e}"
                )
                break

    net.save("models/model.npz")

    X_tf = xs.reshape(-1, 1).astype(np.float32)
    y_tf = ys_true.astype(np.float32)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse"
    )

    history = model.fit(
        X_tf,
        y_tf,
        epochs=200,
        batch_size=32,
        verbose=1
    )

    final_loss = model.evaluate(X_tf, y_tf, verbose=0)
    print(f"* TENSORFLOW FINAL TRAIN LOSS: {final_loss:.6f}")

    model.save("models/tf_model.keras")
