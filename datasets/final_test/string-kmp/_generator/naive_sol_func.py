def solve(T, P):
    n = len(T)
    m = len(P)
    if m == 0:
        return []

    res = []
    for i in range(n - m + 1):
        if T[i:i+m] == P:
            res.append(str(i))

    print(' '.join(res) if res else '')
