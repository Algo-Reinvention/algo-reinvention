import sys

def solve(s):
    if not s:
        print(0)
        return
    
    # Transform S into T
    # S = "abba" -> T = "^#a#b#b#a#$"
    T = '#'.join(f'^{s}$')
    n = len(T)
    P = [0] * n
    C = 0
    R = 0
    
    for i in range(1, n-1):
        P[i] = (R > i) and min(R - i, P[2*C - i])
        # Attempt to expand palindrome centered at i
        while T[i + 1 + P[i]] == T[i - 1 - P[i]]:
            P[i] += 1
        
        # If palindrome centered at i expands past R,
        # adjust center based on expanded palindrome.
        if i + P[i] > R:
            C = i
            R = i + P[i]
    
    # Find the maximum element in P.
    max_len = 0
    for length in P:
        if length > max_len:
            max_len = length
    print(max_len)

def main():
    try:
        s = sys.stdin.read().strip()
        solve(s)
    except Exception:
        pass

if __name__ == '__main__':
    main()
