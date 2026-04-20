def solve(text, pattern):
    n = len(text)
    m = len(pattern)
    res = []
    i = n - 1
    while i >= 0:
        if text[i] == pattern[-1]:
            j = i
            k = m - 1
            while k >= 0 and j >= 0 and text[j] == pattern[k]:
                j -= 1
                k -= 1
            if k < 0:
                res.append(i - m + 1)
            i = j
        else:
            i -= 1
    print(' '.join(map(str, res[::-1])))
