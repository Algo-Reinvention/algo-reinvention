import sys

def compute_lps(pattern):
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps

def solve(text, pattern):
    if not pattern:
        print()
        return

    n = len(text)
    m = len(pattern)
    lps = compute_lps(pattern)

    i = 0
    j = 0
    indices = []
    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1

        if j == m:
            # Found pattern at index i - j
            indices.append(str(i - j))
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    print(" ".join(indices))

def main():
    try:
        lines = sys.stdin.read().splitlines()
        # Filter out empty lines just in case
        lines = [l for l in lines if l]
        if len(lines) >= 2:
            text = lines[0]
            pattern = lines[1]
            solve(text, pattern)
    except Exception:
        pass

if __name__ == '__main__':
    main()
