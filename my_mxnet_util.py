# This file contains utility functions to make life easier
import mxnet as mx
from mxnet import np, npx
npx.set_np()

def try_gpu(i=0):
    """
    Return gpu(i) if it exists, else return cpu()
    """
    return npx.gpu(i) if npx.num_gpus()>=i+1 else npx.cpu()


def synthetic_regression_data(w, b, num_examples):
    """Used to make synthetic data on the run

    Args:
        w (list): list of weights for independent variables
        b (int): some value for bias
        num_examples (int): number of observations

    Returns 
    X, Y (independent data, target data): synthetic data and target data
    """

    X = np.random.normal(0,1,(num_examples, len(w)))
    y = np.dot(X, w) + b
    y+= np.random.normal(0, 0.01, y.shape)
    return X, y.reshape((-1,1))
    
