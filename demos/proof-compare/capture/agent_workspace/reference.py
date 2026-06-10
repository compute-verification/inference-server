"""Reference linear algebra on plain Python lists (no numpy).

Naive O(n^3) matrix product and O(n^2) matrix-vector product. The point of the
task is that freivalds_check verifies a product using ONLY mat_vec (O(n^2) per
round), while matmul is the expensive operation being checked.
"""


def matmul(A, B):
    """Naive matrix product of A (m×k) and B (k×n) -> m×n."""
    k = len(B)
    n = len(B[0])
    return [[sum(row[t] * B[t][j] for t in range(k)) for j in range(n)]
            for row in A]


def mat_vec(M, v):
    """Matrix-vector product M·v on plain lists."""
    return [sum(row[j] * v[j] for j in range(len(v))) for row in M]
