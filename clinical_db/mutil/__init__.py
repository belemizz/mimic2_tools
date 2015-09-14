"""Utility classes and functions."""


from .mycsv import Csv
from .graph import Graph
from .cache import Cache
from .stopwatch import Stopwatch
from .type import is_number, is_number_list, include_any_number, float_list, float_div
__all__ = [Csv, Graph, Cache, Stopwatch,
           is_number, is_number_list, include_any_number, float_list, float_div]


def p_info(word):
    print '[INFO] ' + word


def intersection(list_of_list):
    for index, l in enumerate(list_of_list):
        if index == 0:
            s = set(l)
        else:
            s = s & set(l)
    return list(s)
