#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace funi {

using Index = std::int64_t;

/// Raw storage that is *not* zero-initialized (unlike std::vector), for
/// trivial types only.
template<typename T>
using Storage = std::unique_ptr<T[]>;

template<typename T>
Storage<T> Allocate(const std::size_t n) {
  return Storage<T>(new T[n]); // default-init: no zeroing for trivial T
}

/// Result of `UniqueRows`.
struct Unique {
  Index n_rows{};
  Index n_unique{};
  Storage<Index> ids;     ///< [n_unique] row of the input of each unique row
  Storage<Index> inverse; ///< [n_rows] unique row number of each input row
};

/// np.unique(axis=0)-like routine for float rows with a tolerance.
///
/// Two rows are the same if their euclidean distance is <= `tolerance`.
/// Grouping is greedy: rows are visited in order of their projection onto a
/// fixed generic unit direction, and each not-yet-assigned row claims all
/// unassigned rows within `tolerance` of itself.
///
/// Sorting on a unit-vector projection is exact: |u.d| <= |d|, so no row
/// within tolerance can lie outside the sweep window. A generic direction
/// (rather than a coordinate axis) separates rows that share coordinates, as
/// on grids, so the window stays small.
/// The sorted (key, row) pairs are contiguous, so the sweep reads memory
/// linearly; only the final distance check touches the original rows.
///
/// The representative of each group is its lowest row index, and unique rows
/// are returned in order of first appearance: [1 2 3 3 5 2 4] -> [1 2 3 5 4].
///
/// @param data row-major, `n_rows * n_cols` values
template<typename T>
Unique UniqueRows(const T* data,
                  const Index n_rows,
                  const Index n_cols,
                  const T tolerance) {
  Unique out;
  out.n_rows = n_rows;
  out.inverse = Allocate<Index>(n_rows);
  Index* const inverse = out.inverse.get();
  std::fill_n(inverse, n_rows, Index{-1});

  if (n_rows == 0) {
    return out;
  }
  if (n_cols == 0) { // zero-width rows are all equal
    std::fill_n(inverse, n_rows, Index{0});
    out.n_unique = 1;
    out.ids = Allocate<Index>(1);
    out.ids[0] = 0;
    return out;
  }

  // Fixed, generic unit direction (golden-angle cosines, reproducible). Any
  // unit vector keeps the sweep exact; a generic one gives distinct keys to
  // rows that share coordinates (grids, planes), keeping the window small.
  auto direction = Allocate<double>(n_cols);
  double norm2{};
  for (Index c{}; c < n_cols; ++c) {
    direction[c] = std::cos(1.0 + 2.399963229728653 * static_cast<double>(c));
    norm2 += direction[c] * direction[c];
  }
  for (Index c{}; c < n_cols; ++c) {
    direction[c] /= std::sqrt(norm2);
  }

  // (key, row) sorted by key, then row. Keys are doubles even for float data
  // so that key rounding cannot break the window guarantee.
  struct Keyed {
    double key;
    Index row;
  };
  auto keyed = Allocate<Keyed>(n_rows);
  for (Index r{}; r < n_rows; ++r) {
    double key{};
    for (Index c{}; c < n_cols; ++c) {
      key += direction[c] * static_cast<double>(data[r * n_cols + c]);
    }
    keyed[r] = {key, r};
  }
  std::sort(keyed.get(),
            keyed.get() + n_rows,
            [](const Keyed& a, const Keyed& b) {
              return a.key < b.key || (a.key == b.key && a.row < b.row);
            });

  const T tol2 = tolerance * tolerance;
  auto close = [&](const Index a, const Index b) {
    T sum{};
    for (Index c{}; c < n_cols; ++c) {
      const T d = data[a * n_cols + c] - data[b * n_cols + c];
      sum += d * d;
    }
    return sum <= tol2;
  };

  // greedy sweep; lowest[g] is the lowest row of group g
  auto lowest = Allocate<Index>(n_rows);
  Index n_groups{};
  for (Index s{}; s < n_rows; ++s) {
    const Index first = keyed[s].row;
    if (inverse[first] != -1) {
      continue;
    }
    const Index group = n_groups++;
    Index low = first;
    inverse[first] = group;
    for (Index t = s + 1;
         t < n_rows && keyed[t].key - keyed[s].key <= tolerance;
         ++t) {
      const Index row = keyed[t].row;
      if (inverse[row] == -1 && close(first, row)) {
        inverse[row] = group;
        low = std::min(low, row);
      }
    }
    lowest[group] = low;
  }
  keyed.reset();

  // relabel groups by first appearance: a row opens its group iff it is the
  // group's lowest row
  auto label = Allocate<Index>(n_groups);
  out.n_unique = n_groups;
  out.ids = Allocate<Index>(n_groups);
  Index next{};
  for (Index r{}; r < n_rows; ++r) {
    const Index g = inverse[r];
    if (lowest[g] == r) {
      label[g] = next;
      out.ids[next++] = r;
    }
  }
  for (Index r{}; r < n_rows; ++r) {
    inverse[r] = label[inverse[r]];
  }

  return out;
}

} // namespace funi
