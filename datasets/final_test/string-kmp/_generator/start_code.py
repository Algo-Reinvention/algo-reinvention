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
    
    lines = sys.stdin.read().splitlines()
    lines = [l for l in lines if l]
    if len(lines) >= 2:
        text = lines[0]
        pattern = lines[1]
        solve(text, pattern)
