import sys
import io
import collections
import heapq
import bisect
import math
import cmath
import decimal
import fractions
import statistics
import operator
import itertools
import functools
import re
import string
import copy
import array
import time
import builtins

orig_import = builtins.__import__

def secure_import(name, *args, **kwargs):
    if name == 'random':
        raise ImportError("Forbidden: random is not allowed")
    return orig_import(name, *args, **kwargs)

builtins.__import__ = secure_import

sys.setrecursionlimit(200000)
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


def main():
    line = sys.stdin.readline()
    if not line:
        return

    try:
        n = int(line.strip())
    except ValueError:
        return

    if n < 0:
        return

    solve(n)
