"""Seeded xorshift64 generator -- deterministic randomness for replayable runs.

Use this instead of the `random` module so two runs with the same seed perform
the identical computation, bit for bit.
"""

_MASK = (1 << 64) - 1


class Xorshift:
    def __init__(self, seed=1):
        self.state = (seed & _MASK) or 1

    def next(self):
        x = self.state
        x = (x ^ (x << 13)) & _MASK
        x = x ^ (x >> 7)
        x = (x ^ (x << 17)) & _MASK
        self.state = x
        return x

    def randbit(self):
        """One fair random bit."""
        return self.next() & 1
