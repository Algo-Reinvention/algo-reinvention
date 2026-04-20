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
    if name == 'numpy':
        raise ImportError("Forbidden: numpy is not allowed")
    return orig_import(name, *args, **kwargs)

builtins.__import__ = secure_import

sys.setrecursionlimit(200000)


def main():
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return

    ptr = 0
    n = data[ptr]
    ptr += 1
    if n <= 0:
        return

    total = n * n
    if len(data) < 1 + total + total:
        return

    flat_a = data[ptr:ptr + total]
    ptr += total
    flat_b = data[ptr:ptr + total]

    a = [flat_a[i * n:(i + 1) * n] for i in range(n)]
    b = [flat_b[i * n:(i + 1) * n] for i in range(n)]

    solve(n, a, b)
