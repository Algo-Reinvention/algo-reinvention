import sys

def pre_compute_bc(pattern):
    """Precompute the bad-character table: the last position of each character."""
    m = len(pattern)
    bc = {}
    for i in range(m):
        bc[pattern[i]] = i
    return bc

def pre_compute_gs(pattern):
    """Precompute the good-suffix table."""
    m = len(pattern)
    suffix = [0] * m
    bmGs = [m] * m

    # Compute the suffix array.
    # suffix[i] is the maximum suffix-match length for the substring ending at i.
    suffix[m - 1] = m
    g = m - 1
    f = 0
    for i in range(m - 2, -1, -1):
        if i > g and suffix[i + m - 1 - f] < i - g:
            suffix[i] = suffix[i + m - 1 - f]
        else:
            if i < g:
                g = i
            f = i
            while g >= 0 and pattern[g] == pattern[g + m - 1 - f]:
                g -= 1
            suffix[i] = f - g

    # Case 1: a substring in the pattern matches the good suffix exactly.
    # Case 2: a prefix of the pattern matches a suffix of the good suffix (border case).

    # Handle case 2.
    j = 0
    for i in range(m - 1, -1, -1):
        if suffix[i] == i + 1:
            while j < m - 1 - i:
                if bmGs[j] == m:
                    bmGs[j] = m - 1 - i
                j += 1

    # Handle case 1.
    for i in range(m - 1):
        bmGs[m - 1 - suffix[i]] = m - 1 - i

    return bmGs

def solve(text, pattern):
    if not pattern:
        print("")
        return

    m = len(pattern)
    n = len(text)
    if m > n:
        print("")
        return

    # Precompute shift tables.
    bc = pre_compute_bc(pattern)
    bmGs = pre_compute_gs(pattern)

    indices = []
    s = 0  # s is the starting offset of the pattern relative to the text.
    while s <= n - m:
        j = m - 1
        # Match from right to left.
        while j >= 0 and pattern[j] == text[s + j]:
            j -= 1

        if j < 0:
            # Full match found.
            indices.append(str(s))
            # After a full match, shift according to the good-suffix rule (bmGs[0]).
            s += bmGs[0]
        else:
            # On mismatch, take the larger shift from the bad-character and good-suffix rules.
            char_at_text = text[s + j]
            # Bad-character rule: shift = mismatch index - last occurrence in the pattern.
            bc_shift = j - bc.get(char_at_text, -1)
            gs_shift = bmGs[j]
            s += max(bc_shift, gs_shift)

    print(" ".join(indices))

# Example entry point.
if __name__ == "__main__":
    # Input can be read from sys.stdin if needed.
    # line1 = sys.stdin.readline().strip()
    # line2 = sys.stdin.readline().strip()
    # solve(line1, line2)
    pass
