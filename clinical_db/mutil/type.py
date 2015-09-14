import numpy as np


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


def float_list(l, fill=0.):
    """Comvert list into float list
    :param l: list to convert
    :param fill: fill this value when conversion is not possible
    """
    f_list = []
    for s in l:
        try:
            f_list.append(float(s))
        except ValueError:
            f_list.append(0.)

    return np.array(f_list)


def float_div(a, b):
    try:
        return float(a) / b
    except ZeroDivisionError:
        return 0.
