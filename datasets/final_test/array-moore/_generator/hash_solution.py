def solve(n, a):
    """
    Find the majority element using a Hash Map.
    :param n: Number of elements
    :param a: List of integers
    """
    # Initialize the hash map (a dictionary in Python).
    counts = {}
    
    # Set the threshold for the majority element.
    threshold = n // 2
    
    for x in a:
        # Update the count of the current element.
        # get(x, 0) returns the current count, or 0 if x is not present.
        counts[x] = counts.get(x, 0) + 1
        
        # Check online: once an element exceeds n/2, it must be the majority element.
        if counts[x] > threshold:
            print(x)
            return

    # You could also use Python's built-in Counter for a shorter implementation:
    # from collections import Counter
    # counts = Counter(a)
    # for x, count in counts.items():
    #     if count > n // 2:
    #         print(x)
    #         return
