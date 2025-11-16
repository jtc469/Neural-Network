import numpy as np

class Activation:
    def __call__(self, x):
        x = np.asarray(x)
        return self.func(x)

    def func(self, x):
        raise NotImplementedError


class ReLU(Activation):
    def func(self, x):
        return np.maximum(0, x)
    
    def d(self, x):
        return (x > 0).astype(float)


class Sigmoid(Activation):
    def func(self, x):
        return 1 / (1 + np.exp(-x))

    def d(self, x):
        s = self.__call__(x)
        return s * (1 - s)
    
class Linear:
    def __call__(self, x):
        return x

    def d(self, x):
        return np.ones_like(x)
