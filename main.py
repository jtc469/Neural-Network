import test 
import train
import os 
import time


# Retrain with del models/model.npz, or change path
model_path = "models/model.npz"


if not os.path.exists(model_path):
    print("==========================")
    print("MODEL TRAINING IN PROGRESS")
    start = time.time()
    train.train()
    stop = time.time()
    print(f"Time Taken: {(stop-start):.4f}s")
    print("========================== \n")

test.test()