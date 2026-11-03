import unittest
import warnings

import numpy as np

import funi  # isort:skip


def check(q, tol):
    data, ids, inv = funi.unique_rows(q, tol)
    assert np.array_equal(data, q[ids])
    assert np.all(np.linalg.norm(q - data[inv], axis=1) <= tol)
    assert np.all(np.diff(ids) > 0)  # order of appearance
    return data, ids, inv


class FuniTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        q = rng.random((100, 3))
        self.q = np.vstack((q, q, q))
        rng.shuffle(self.q)

    def test_duplicates(self):
        for dtype in (np.float64, np.float32):
            data, _, _ = check(self.q.astype(dtype), 1e-6)
            assert data.dtype == dtype
            assert len(data) == 100

    def test_order_of_appearance(self):
        x = np.array([1, 2, 3, 3, 5, 2, 4], dtype=float)[:, None]
        data, ids, inv = funi.unique_rows(x, 1e-9)
        assert data.ravel().tolist() == [1, 2, 3, 5, 4]
        assert ids.tolist() == [0, 1, 2, 4, 6]
        assert inv.tolist() == [0, 1, 2, 2, 3, 1, 4]

    def test_first_occurrence(self):
        _, ids, _ = funi.unique_rows(self.q, 1e-10)
        for i in ids:
            assert not np.any(np.all(self.q[:i] == self.q[i], axis=1))

    def test_noise_within_tolerance(self):
        # diagonal offsets are the hard case for projection based methods
        rng = np.random.default_rng(1)
        base = rng.random((200, 5)) * 10
        noise = rng.normal(size=base.shape)
        noise *= 0.5e-3 / np.linalg.norm(noise, axis=1, keepdims=True)
        q = np.vstack((base, base + noise))
        data, _, _ = check(q, 1e-3)
        assert len(data) == 200

    def test_grid(self):
        # many rows share every coordinate; must stay fast and exact
        g = np.arange(60, dtype=float)
        grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
        q = np.vstack((grid, grid + 1e-9))
        data, _, _ = check(q, 1e-6)
        assert len(data) == len(grid)

    def test_degenerate_axis(self):
        # all points identical along x, anti-diagonal in y/z
        t = np.linspace(0, 1, 500)
        q = np.stack((np.zeros(500), t, 1 - t), axis=1)
        data, _, _ = check(np.vstack((q, q)), 1e-9)
        assert len(data) == 500

    def test_edge_cases(self):
        for shape in [(0, 3), (1, 3), (5, 1)]:
            check(np.random.rand(*shape), 1e-9)
        data, ids, inv = funi.unique_rows(np.zeros((4, 2)), 0.0)
        assert len(data) == 1 and inv.tolist() == [0] * 4

    def test_non_contiguous_and_errors(self):
        check(self.q[:, ::-1], 1e-9)
        check(np.asfortranarray(self.q), 1e-9)
        with self.assertRaises(ValueError):
            funi.unique_rows(np.zeros(3), 1e-9)
        with self.assertRaises(ValueError):
            funi.unique_rows(np.zeros((3, 3), dtype=int), 1e-9)

    def test_deprecated_arguments(self):
        for args in [(True, "l"), (False, "axis"), (True,)]:
            with self.assertWarns(DeprecationWarning):
                data, ids, inv = funi.unique_rows(self.q, 1e-9, *args)
            assert len(data) == 100 and np.all(np.diff(ids) > 0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            funi.unique_rows(self.q, 1e-9)  # no warning without them


if __name__ == "__main__":
    unittest.main()
