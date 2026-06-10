# Workspace: implement Freivalds' check

Task for the coding agent:

Read the mini-paper in `paper.md` and implement Freivalds' randomized check
for matrix products.

Deliverables:

- `freivalds.py` — `freivalds_check(A, B, C, k=16, seed=1)` returning `True`
  iff all k rounds accept. Randomness MUST come from `rng.Xorshift(seed)`
  (deterministic replay; see the paper's reproducibility note). Use
  `reference.mat_vec` for the matrix–vector products.
- `test_freivalds.py` — `unittest` tests covering: a correct product is
  accepted; a corrupted product (one entry changed) is rejected; the check is
  deterministic for a fixed seed.

Existing modules: `reference.py` (matmul, mat_vec), `rng.py` (Xorshift).
The tests are run with `python3 -m unittest discover` in this directory.
