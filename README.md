# my_mx_util
A bunch of everyday functions to make life easier


```python
# importing libraries
import mxnet as mx
from mxnet import np, npx
import time
from my_mx_util import *
npx.set_np()
```

## try_gpu()
allocate gpu if present else go with cpu


```python
def try_gpu(i=0):
    """
    Return gpu(i) if it exists, else return cpu()
    """
    return npx.gpu(i) if npx.num_gpus()>=i+1 else npx.cpu()
```


```python
# allocating cpu
x = np.random.uniform(size=(0), ctx= npx.cpu())

# allocating gpu
y = np.random.uniform(size=(0), ctx= npx.gpu())

# try allocating gpu if it is available, else cpu
z = np.random.uniform(size=(0), ctx= try_gpu())

x,'    ', y,'    ', z
```




    (array([]), '    ', array([], ctx=gpu(0)), '    ', array([], ctx=gpu(0)))



## synthetic_data() (regression)
this function creates synthetic data for quick testing


```python
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
```


```python
# defining variables
weights = np.array([2,-3])
bias = 5
observations = 10

# creating synthetic data
features, targets = synthetic_regression_data(weights, bias, observations)
print('features\n', features)
print('targets\n', targets)
```

    features
     [[ 2.2122064   1.1630787 ]
     [ 0.7740038   0.4838046 ]
     [ 1.0434403   0.29956347]
     [ 1.1839255   0.15302546]
     [ 1.8917114  -1.1688148 ]
     [-1.2347414   1.5580711 ]
     [-1.771029   -0.5459446 ]
     [-0.45138445 -2.3556297 ]
     [ 0.57938355  0.5414402 ]
     [-1.856082    2.6785066 ]]
    targets
     [[ 5.915408 ]
     [ 5.1091404]
     [ 6.1861105]
     [ 6.9032865]
     [12.292312 ]
     [-2.150507 ]
     [ 3.095404 ]
     [11.162766 ]
     [ 4.529569 ]
     [-6.7439113]]

