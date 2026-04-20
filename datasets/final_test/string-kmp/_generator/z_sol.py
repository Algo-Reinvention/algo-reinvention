import sys

def compute_z(s):
    n = len(s)
    z = [0] * n
    l, r = 0, 0
    for i in range(1, n):
        if i <= r:
            z[i] = min(r - i + 1, z[i - l])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] - 1 > r:
            l, r = i, i + z[i] - 1
    return z

def solve(text, pattern):
    if not pattern:
        print()
        return

    m = len(pattern)
    n = len(text)
    if m > n:
        print("")
        return

    # Build an auxiliary string: Pattern + separator + Text.
    # The separator must not appear in either pattern or text.
    concat = pattern + "$" + text
    z = compute_z(concat)

    indices = []
    # Start checking from position m + 1, which is the start of the text region.
    for i in range(m + 1, len(concat)):
        if z[i] == m:
            indices.append(str(i - (m + 1)))

    # print(" ".join(indices))

# The main entry point would be similar to the KMP version.
