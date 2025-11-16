import numpy as np

class Loss:
    def __call__(self, pred, target):
        return self.func(pred, target)
    
class MSE(Loss):
    def func(self, pred, target):
        return np.mean((pred - target) ** 2)
    
    def d(self, pred, target):
        return 2 * (pred - target) / target.size