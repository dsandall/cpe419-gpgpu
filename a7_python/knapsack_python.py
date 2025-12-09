import cupy as cp
import numpy as np
import time


# ---------------------------------------
# FRACTIONAL KNAPSACK
# ---------------------------------------
# solution:
# - compute densities in parallel
# - sort by value density (also parallel)
# - assuming greedy algorithm, transform sorted list of weights into prefix sum list
# - this creates a monotonically increasing list which can be efficiently searched for the highest non-fractional solution
# - then take fractional component of next highest density item.
# - (i do this in video games all the time... might be handy... hmmm) ---------------------------------------
# ---------------------------------------


def fractional_knapsack_cuda(values, weights, capacity):
    n = len(values)

    # Move arrays to GPU
    v = cp.asarray(values, dtype=cp.float32)
    w = cp.asarray(weights, dtype=cp.float32)

    # value density
    density = v / w

    # sort by density descending
    idx = cp.argsort(-density)
    v = v[idx]
    w = w[idx]

    # prefix sum of weights on GPU
    prefix = cp.cumsum(w)

    # Bring prefix to CPU to find cutoff
    prefix_cpu = prefix.get()

    # Find the first index where weight exceeds capacity
    cutoff = np.searchsorted(prefix_cpu, capacity)

    total_value = 0.0
    total_weight = 0.0

    # Take full items up to cutoff−1
    for i in range(min(cutoff, n)):
        if prefix_cpu[i] <= capacity:
            total_value += float(v[i])
            total_weight += float(w[i])

    # Fraction from the cutoff item
    if cutoff < n:
        remaining = capacity - total_weight
        if remaining > 0:
            frac = remaining / float(w[cutoff])
            total_value += float(v[cutoff]) * frac

    return total_value


def runner(v, w, c):
    start = time.perf_counter()
    res = fractional_knapsack_cuda(v, w, c)
    end = time.perf_counter()
    print(f"Max: {res} , Elapsed: {end - start:.6f} seconds")


# ---------------------------------------
# entry point:
# ---------------------------------------
values = np.array([60, 100, 120], dtype=np.float32)
weights = np.array([10, 20, 30], dtype=np.float32)
capacity = 50
runner(values, weights, capacity)

np.random.seed(42)

values_500 = np.random.randint(1, 101, size=500).astype(np.float32)
weights_500 = np.random.randint(1, 50, size=500).astype(np.float32)
capacity_500 = 10000.0
runner(values_500, weights_500, capacity_500)

values_1000 = np.random.randint(1, 101, size=5000000).astype(np.float32)
weights_1000 = np.random.randint(1, 50, size=5000000).astype(np.float32)
capacity_1000 = 600000.0
runner(values_1000, weights_1000, capacity_1000)
