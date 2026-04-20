def solve(n, a):
    """
    Find the majority element using Sorting.
    :param n: Number of elements
    :param a: List of integers
    """
    # 1. Sort the array in place.
    # Python's sort() uses Timsort with average and worst-case complexity O(n log n).
    a.sort()
    
    # 2. After sorting, the majority element must sit at the middle position.
    # For an array of length n, index n // 2 is the midpoint.
    majority_element = a[n // 2]
    
    # Output the result.
    print(majority_element)
