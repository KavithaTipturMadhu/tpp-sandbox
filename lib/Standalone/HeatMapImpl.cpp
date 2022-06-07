//===- HeatMapImpl.cpp - ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/HeatMap.h"

#include <array>
#include <cassert>
#include <map>

using namespace mlir;
using namespace mlir::tpp::x86;

extern uint64_t kHeatmapSize;
extern uint16_t kHeatmapKeys[][3];
extern float kHeatmapValues[];

static const unsigned kStartupCost = 32;

using Tile = std::tuple<unsigned, unsigned, unsigned>;

static std::array<Tile, 1> specialStencils = {
    Tile{64, 16, 3},
};

struct Heatmap {
  Heatmap() {
    for (unsigned i = 0; i < kHeatmapSize; ++i) {
      byTile.emplace(
          Tile{kHeatmapKeys[i][0], kHeatmapKeys[i][1], kHeatmapKeys[i][2]},
          kHeatmapValues[i]);
    }
  }

  std::map<Tile, double> byTile;
};

static Heatmap heatmap;

KernelCost heatmapCost(llvm::ArrayRef<int64_t> ranges) {
  assert(ranges.size() == 3 && "heatmapCost expects a 3D tile");

  auto tile = Tile{ranges[0], ranges[1], ranges[2]};
  auto it = heatmap.byTile.find(tile);
  if (it != heatmap.byTile.end()) {
    return KernelCost{it->second, kStartupCost};
  }

  // We mainly care about M and K. If both (m, n - 1, k) and (m, n + 1, k)
  // exist, we may use their average value for prediction.
  auto itLower = heatmap.byTile.find(Tile{ranges[0], ranges[1] - 1, ranges[2]});
  if (ranges[1] == 1 || itLower != heatmap.byTile.end()) {
    auto itUpper =
        heatmap.byTile.find(Tile{ranges[0], ranges[1] + 1, ranges[2]});
    if (itUpper != heatmap.byTile.end()) {
      auto throughput = (ranges[1] > 1)
                            ? ((itLower->second + itUpper->second) / 2)
                            : itUpper->second;
      return KernelCost{throughput, kStartupCost};
    }
  }

  // If we cannot find (m, n, k) in the heatmap, try the special cases.
  for (auto stencil : specialStencils) {
    if (stencil == tile) {
      return KernelCost{0.001, kStartupCost};
    }
  }

  return KernelCost{0.0, 0};
}
