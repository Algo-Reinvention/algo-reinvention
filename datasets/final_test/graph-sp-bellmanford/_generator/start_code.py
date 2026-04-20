import sys
import io
import collections
import heapq
import bisect
import math
import cmath
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


sys.setrecursionlimit(200000)

def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return

    it = iter(data)

    n = int(next(it))
    m = int(next(it))

    graph = [[] for _ in range(n + 1)]

    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        w = int(next(it))
        graph[u].append((v, w))

    s = int(next(it))

    solve(n, m, graph, s)
