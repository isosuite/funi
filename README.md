# funi
Find UNIque float array rows.
[numpy.unique](https://numpy.org/doc/stable/reference/generated/numpy.unique.html) with `axis=0`, but for float arrays and with a tolerance.

Rows are sorted by their projection onto a fixed generic unit direction, then swept: each row claims every
not-yet-claimed row within `tolerance` (euclidean distance). The sweep window is exact, i.e. no duplicate is
missed. It is fast whenever few rows project to nearly the same key, which a generic direction ensures even
for grids; it only degrades if many distinct rows lie within `tolerance` of one hyperplane orthogonal to it.

## Install
```bash
pip install funi
```

## Quick Start
```python
import funi
import numpy as np

arr = np.random.random((10000, 3))
arr = np.vstack((arr, arr, arr))
np.random.shuffle(arr)

unique_data, unique_ids, inverse = funi.unique_rows(arr, tolerance=1e-11)

assert np.allclose(unique_data, arr[unique_ids])  # unique rows of the original array
assert np.allclose(arr, unique_data[inverse])     # map back to the original array
assert np.all(np.diff(unique_ids) > 0)            # first occurrences, in order of appearance
```


`unique_ids` are the first occurrence of each group; unique rows keep their order of appearance (`[1 2 3 3 5 2 4]` -> `[1 2 3 5 4]`).
