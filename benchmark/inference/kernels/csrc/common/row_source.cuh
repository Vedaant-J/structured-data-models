// Source aspect for CUTLASS 2.x epilogues whose operator needs the output coordinates. Like the stock aspect it
// streams the C operand (here the bias, stride 0); it also records where each output fragment starts.
#pragma once
#include <cutlass/cutlass.h>
#include <cutlass/matrix_coord.h>

namespace row_source {
template <class Iterator>
struct BiasSource {
  Iterator bias_iterator, coordinate_iterator;
  typename Iterator::Fragment bias;
  cutlass::MatrixCoord start;
  CUTLASS_DEVICE void load() {
    bias_iterator.load(bias);
    start = coordinate_iterator.thread_start();
    ++bias_iterator;
    ++coordinate_iterator;
  }
};

// Calls f(row, fragment_row) for each output row of the fragment that starts at `start`, in fragment order.
template <class Map, class F>
CUTLASS_DEVICE void for_each_row(cutlass::MatrixCoord start, F&& f) {
  CUTLASS_PRAGMA_UNROLL
  for (int cluster = 0; cluster < Map::Iterations::kCluster; ++cluster) {
    CUTLASS_PRAGMA_UNROLL
    for (int group = 0; group < Map::Iterations::kGroup; ++group) {
      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < Map::Iterations::kRow; ++r)
        f(start.row() + r * Map::Delta::kRow + group * Map::Delta::kGroup + cluster * Map::Delta::kCluster,
          r + Map::Iterations::kRow * (group + Map::Iterations::kGroup * cluster));
    }
  }
}
}  // namespace row_source
