import numpy as np
from network import Network, Layer
from activations import ReLU, Sigmoid, Linear
from losses import MSE



def train():
    relu = ReLU()
    sig = Sigmoid()
    linear = Linear()
    loss_fn = MSE()

    loss_tol = 1e-3 # How small loss must get before quitting

    plateau_tol = 1e-5
    repeat_loss = 500 # how many epochs must plateau before quitting 
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
    epochs = 4_000

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
            print(f"* TRAINING: {epoch}/{epochs} (LOSS: {loss:.6f})")

        # stop if loss < tolerance (effectively loss->0)
        if loss < loss_tol:
            print(f"EARLY STOP (tolerance) at epoch {epoch}, loss {loss:.6f}")
            break

        # stop if losses plateau
        if len(losses) >= repeat_loss:
            recent = losses[-repeat_loss:]
            if max(recent) - min(recent) < plateau_tol:
                print(
                    f"EARLY STOP (plateau) at epoch {epoch}, loss {loss:.6f} \n"
                    f"Loss range {max(recent) - min(recent):.6e}"
                )
                break

    net.save("models/model.npz")