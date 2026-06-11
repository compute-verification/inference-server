# Freivalds' Check: Probabilistic Verification of Matrix Products

## Abstract

Verifying that C = A·B by recomputing the product costs O(n^3) multiplications
(naively). Freivalds (1977) showed the product can be *checked* in O(n^2) per
round with one-sided error: a correct product is always accepted, and an
incorrect product is accepted with probability at most 1/2 per round, so k
independent rounds drive the error below 2^-k. This is the foundation of
cheap compute attestation: a verifier can audit a prover's matmul at a
quadratic, not cubic, cost.

## Method

Let A, B, C be n×n integer matrices. Each round:

1. Draw a random vector r ∈ {0,1}^n (each entry an independent fair bit).
2. Compute x = B·r            (one matrix–vector product, O(n^2))
3. Compute y = A·x = A·(B·r)  (one matrix–vector product, O(n^2))
4. Compute z = C·r            (one matrix–vector product, O(n^2))
5. If y ≠ z, output REJECT (the product is certainly wrong).

If all k rounds pass, output ACCEPT.

## Analysis

If C = A·B then y = A·B·r = C·r = z for every r, so the check never falsely
rejects. If C ≠ A·B, let D = A·B − C ≠ 0. The check passes only when D·r = 0.
Fix a row of D with a nonzero entry d_j. Conditioning on the other coordinates
of r, at most one of the two choices of r_j makes the row's dot product zero,
so Pr[D·r = 0] ≤ 1/2. Rounds are independent, hence Pr[accept wrong C] ≤ 2^-k.

## Reproducibility note

For deterministic replay, draw the random bits from a seeded xorshift
generator rather than an entropy source; two verifiers with the same seed then
perform the identical check, bit for bit.
