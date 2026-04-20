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
    line1 = sys.stdin.readline().strip()
    if not line1: return
    n = int(line1)
    
    line2 = sys.stdin.readline().strip()
    if not line2: return
    m = int(line2)
    
    INF = float('inf')
    dist = [[INF] * (n + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        dist[i][i] = 0
        
    for _ in range(m):
        u, v, w = map(int, sys.stdin.readline().split())
        if w < dist[u][v]:
            dist[u][v] = w
            
    solve(n, dist)
