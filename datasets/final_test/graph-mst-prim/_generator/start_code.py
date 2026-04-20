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

'''
YOUR CODE IS HERE
'''

def main():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    ptr = 0
    
    n = int(input_data[ptr])
    ptr += 1
    m = int(input_data[ptr])
    ptr += 1
    
    adj = [[] for _ in range(n + 1)]
    
    for _ in range(m):
        if ptr + 2 >= len(input_data):
            break
        u = int(input_data[ptr])
        v = int(input_data[ptr + 1])
        w = int(input_data[ptr + 2])
        ptr += 3
        
        adj[u].append((v, w))
        adj[v].append((u, w))
        
    solve(n, m, adj)
