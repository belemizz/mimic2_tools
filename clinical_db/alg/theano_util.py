'''Utility functions for theano'''
import numpy as np
import theano
import theano.tensor as T


def convert_to_tensor_shared_variable(set_x):
    """convert to the shared valuable"""
    if type(set_x) is T.sharedvar.TensorSharedVariable:
        shared_x = set_x
    elif type(set_x) is np.ndarray:
        shared_x = theano.shared(np.asarray(set_x, dtype=theano.config.floatX), borrow=True)
    else:
        raise TypeError("Sample set, set_x should be TensorSharedValuable or np.ndarray")
    return shared_x
