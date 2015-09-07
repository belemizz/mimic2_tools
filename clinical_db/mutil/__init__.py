"""Utility classes and functions."""

import numpy as np

from .mycsv import Csv
from .graph import Graph
from .cache import Cache
from .stopwatch import Stopwatch
__all__ = [Csv, Graph, Cache, Stopwatch]


def p_info(word):
    print '[INFO] ' + word


def is_number(s):
    """Check if s is nuber"""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def is_number_list(l):
    """Check if list l is all numbers"""
    for s in l:
        if not is_number(s):
            return False
    return True


def include_any_number(l):
    """Check if list l includes any number"""
    for s in l:
        if is_number(s):
            return True
    return False


def float_list(l):
    f_list = []
    for s in l:
        f_list.append(float(s))
    return np.array(f_list)


def float_div(a, b):
    try:
        return float(a) / b
    except ZeroDivisionError:
        return 0.
