import sys
import io
import collections
import heapq
import bisect
import random
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
    if name == 'math':
        raise ImportError("Forbidden: math is not allowed")
    return orig_import(name, *args, **kwargs)

builtins.__import__ = secure_import

sys.setrecursionlimit(200000)

def main():
    # Read two integers a and b from a single line
    line = sys.stdin.readline().strip()
    if not line:
        return

    try:
        parts = list(map(int, line.split()))
        if len(parts) < 2:
            return
        a, b = parts[0], parts[1]
    except ValueError:
        return

    solve(a, b)
