import time

import numpy as np

import funi


def best(f, n=3):
    t = []
    for _ in range(n):
        s = time.perf_counter()
        r = f()
        t.append(time.perf_counter() - s)
    return min(t), r


def cases(n):
    rng = np.random.default_rng(0)
    m = int(round(n ** (1 / 3)))
    g = np.arange(m, dtype=float)
    grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    cloud = rng.random((n, 3))
    line = np.stack((np.linspace(0, 1, n),) * 3, 1)
    for name, base in [("cloud", cloud), ("grid", grid), ("line", line)]:
        q = np.vstack((base, base, base))
        rng.shuffle(q)
        yield name, q, len(base)


print(f"{'case':6} {'rows':>9} {'funi ms':>9} {'np.unique ms':>13}  unique")
for n in (10_000, 100_000, 1_000_000):
    for name, q, n_unique in cases(n):
        tf, (data, _, _) = best(lambda: funi.unique_rows(q, 1e-9))
        tn, ref = best(lambda: np.unique(q, axis=0), n=1 if n > 100_000 else 3)
        assert len(data) == len(ref) == n_unique, (len(data), len(ref))
        print(f"{name:6} {len(q):9} {tf*1e3:9.1f} {tn*1e3:13.1f}  {len(data)}")
