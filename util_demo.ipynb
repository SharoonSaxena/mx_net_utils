{
 "metadata": {
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.2-final"
  },
  "orig_nbformat": 2,
  "kernelspec": {
   "name": "python_defaultSpec_1597224000081",
   "display_name": "Python 3.8.2 64-bit"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2,
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import mxnet as mx\n",
    "from mxnet import np, npx\n",
    "import time\n",
    "from my_mx_util import *\n",
    "npx.set_np()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## try_gpu()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": "(array([]), '    ', array([], ctx=gpu(0)), '    ', array([], ctx=gpu(0)))"
     },
     "metadata": {},
     "execution_count": 2
    }
   ],
   "source": [
    "# allocating cpu\n",
    "x = np.random.uniform(size=(0), ctx= npx.cpu())\n",
    "\n",
    "# allocating gpu\n",
    "y = np.random.uniform(size=(0), ctx= npx.gpu())\n",
    "\n",
    "# try allocating gpu if it is available, else cpu\n",
    "z = np.random.uniform(size=(0), ctx= try_gpu())\n",
    "\n",
    "x,'    ', y,'    ', z"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## synthetic_data() (regression)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": "features\n [[ 2.2122064   1.1630787 ]\n [ 0.7740038   0.4838046 ]\n [ 1.0434403   0.29956347]\n [ 1.1839255   0.15302546]\n [ 1.8917114  -1.1688148 ]\n [-1.2347414   1.5580711 ]\n [-1.771029   -0.5459446 ]\n [-0.45138445 -2.3556297 ]\n [ 0.57938355  0.5414402 ]\n [-1.856082    2.6785066 ]]\ntargets\n [[ 5.915408 ]\n [ 5.1091404]\n [ 6.1861105]\n [ 6.9032865]\n [12.292312 ]\n [-2.150507 ]\n [ 3.095404 ]\n [11.162766 ]\n [ 4.529569 ]\n [-6.7439113]]\n"
    }
   ],
   "source": [
    "# defining variables\n",
    "weights = np.array([2,-3])\n",
    "bias = 5\n",
    "observations = 10\n",
    "\n",
    "# creating synthetic data\n",
    "features, targets = synthetic_regression_data(weights, bias, observations)\n",
    "print('features\\n', features)\n",
    "print('targets\\n', targets)"
   ]
  }
 ]
}
