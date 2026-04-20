import sys
import time

def solve(s):
    # Naive O(N^2)
    n = len(s)
    max_len = 0
    # Center expansion
    # For N=200,000, N^2 is huge. "All a" case triggers worst case.
    
    for i in range(n):
        l, r = i, i
        while l >= 0 and r < n and s[l] == s[r]:
            if r - l + 1 > max_len:
                max_len = r - l + 1
            l -= 1
            r += 1
            
    for i in range(n - 1):
        l, r = i, i + 1
        while l >= 0 and r < n and s[l] == s[r]:
            if r - l + 1 > max_len:
                max_len = r - l + 1
            l -= 1
            r += 1
            
    print(max_len)
