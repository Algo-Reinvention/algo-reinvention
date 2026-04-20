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
    n = int(sys.stdin.readline())
    m = int(sys.stdin.readline())
    graph = [[] for _ in range(n + 1)]
    for _ in range(m):
        u, v, w = map(int, sys.stdin.readline().split())
        graph[int(u)].append((int(v), int(w)))
    s = int(sys.stdin.readline())
    solve(n, m, graph, s)