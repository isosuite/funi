#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "funi.hpp"

namespace nb = nanobind;

namespace {

using Array = nb::ndarray<nb::device::cpu>;

/// Hand raw storage to numpy without copying.
template<typename T>
nb::object ToNumpy(funi::Storage<T> values, std::vector<size_t> shape) {
  T* data = values.get();
  nb::capsule owner(values.release(), [](void* p) noexcept {
    delete[] static_cast<T*>(p);
  });
  return nb::cast(
      nb::ndarray<nb::numpy, T>(data, shape.size(), shape.data(), owner));
}

template<typename T>
nb::tuple Run(const Array& input, const double tolerance) {
  const size_t n_rows = input.shape(0);
  const size_t n_cols = input.shape(1);
  const T* src = static_cast<const T*>(input.data());
  const int64_t row_stride = input.stride(0);
  const int64_t col_stride = input.stride(1);

  // contiguous copy of the input if needed
  funi::Storage<T> copy;
  const T* data = src;
  if (row_stride != static_cast<int64_t>(n_cols) || col_stride != 1) {
    copy = funi::Allocate<T>(n_rows * n_cols);
    for (size_t r{}; r < n_rows; ++r) {
      for (size_t c{}; c < n_cols; ++c) {
        copy[r * n_cols + c] = src[r * row_stride + c * col_stride];
      }
    }
    data = copy.get();
  }

  auto u = funi::UniqueRows<T>(data, n_rows, n_cols, static_cast<T>(tolerance));

  const size_t n_unique = u.n_unique;
  auto unique_data = funi::Allocate<T>(n_unique * n_cols);
  for (size_t i{}; i < n_unique; ++i) {
    std::copy_n(data + u.ids[i] * n_cols, n_cols, &unique_data[i * n_cols]);
  }

  return nb::make_tuple(ToNumpy(std::move(unique_data), {n_unique, n_cols}),
                        ToNumpy(std::move(u.ids), {n_unique}),
                        ToNumpy(std::move(u.inverse), {n_rows}));
}

nb::tuple UniqueRows(const Array& query,
                     const double tolerance,
                     const std::optional<bool>& sorted_index,
                     const std::optional<std::string>& method) {
  // Python's default warning filter shows this once per call site.
  if (sorted_index || method) {
    if (PyErr_WarnEx(PyExc_DeprecationWarning,
                     "funi.unique_rows: `sorted_index` and `method` are "
                     "deprecated and ignored; unique rows are always returned "
                     "in order of first appearance.",
                     2)
        < 0) {
      throw nb::python_error();
    }
  }
  if (query.ndim() != 2) {
    throw std::invalid_argument("input array must be 2D");
  }
  if (query.dtype() == nb::dtype<double>()) {
    return Run<double>(query, tolerance);
  }
  if (query.dtype() == nb::dtype<float>()) {
    return Run<float>(query, tolerance);
  }
  throw std::invalid_argument("funi supports float32 and float64. For integer "
                              "types, use `np.unique(data, axis=0)`");
}

} // namespace

NB_MODULE(funi, m) {
  m.doc() = "Find unique float array rows within a tolerance.";
  m.def("unique_rows",
        &UniqueRows,
        nb::arg("query"),
        nb::arg("tolerance"),
        nb::arg("sorted_index") = nb::none(),
        nb::arg("method") = nb::none(),
        "Returns (unique_rows, unique_ids, inverse). Unique rows are in order "
        "of first appearance. `sorted_index` and `method` are deprecated, "
        "ignored and only kept for backward compatibility.");
  m.attr("__version__") = "0.1.0";
}
