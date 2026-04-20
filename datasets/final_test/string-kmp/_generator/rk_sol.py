def solve(text, pattern):
    n = len(text)
    m = len(pattern)
    if m == 0:
        return
        base = 256
        mod = 10**9 + 7
        # Precompute ord values for text and pattern
        text_ord = [ord(c) for c in text]
        pattern_ord = [ord(c) for c in pattern]
        power = pow(base, m, mod)
        p_hash = 0
        for char in pattern_ord:
            p_hash = (p_hash * base + char) % mod
            t_hash = 0
            for i in range(m):
                t_hash = (t_hash * base + text_ord[i]) % mod
                res = []
                for i in range(n - m + 1):
                    if i == 0:
                        if t_hash == p_hash:
                            res.append(str(0))
                    else:
                        t_hash = (t_hash * base - text_ord[i - 1] * power + text_ord[i + m - 1]) % mod
                        if t_hash == p_hash:
                            res.append(str(i))
                            print(' '.join(res))
