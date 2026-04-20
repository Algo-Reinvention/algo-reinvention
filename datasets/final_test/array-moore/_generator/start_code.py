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

def main():
    line1 = sys.stdin.readline()
    if not line1:
        return
    n = int(line1.strip())

    a = list(map(int, sys.stdin.read().split()))

    if len(a) > 0:
        solve(n, a)
