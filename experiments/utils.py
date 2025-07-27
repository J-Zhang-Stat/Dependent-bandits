import numpy as np

def stats(arr):
    # arr: shape (repeat, T)
    avg = arr.mean(axis=0)
    sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])  # standard error of the mean
    z = 1.96
    lb  = avg - z * sem
    ub  = avg + z * sem
    return avg, ub, lb

