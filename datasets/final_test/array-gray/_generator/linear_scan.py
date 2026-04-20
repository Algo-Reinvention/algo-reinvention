"""
Correct but O(n): scan from 0 to n and keep the final BRGC value.
"""


def solve(n):
    ans = 0
    i = 0
    while i <= n:
        ans = i ^ (i >> 1)
        i += 1
    print(ans)
