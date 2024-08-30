// RUN: tpp-opt %s  --vectorization-pass --split-input-file | FileCheck %s

module {
  func.func @entry(%arg0: memref<2x4x8x2xf32>, %arg1: memref<2x4x2x4xf32>, %arg2: memref<2x2x8x4xf32>) {
    scf.forall (%arg3, %arg4) in (2, 2) {
      %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 4, 8, 2] [1, 1, 1, 1] : memref<2x4x8x2xf32> to memref<4x8x2xf32, strided<[16, 2, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[%arg4, 0, 0, 0] [1, 4, 2, 4] [1, 1, 1, 1] : memref<2x4x2x4xf32> to memref<4x2x4xf32, strided<[8, 4, 1], offset: ?>>
      %subview_1 = memref.subview %arg2[%arg3, %arg4, 0, 0] [1, 1, 8, 4] [1, 1, 1, 1] : memref<2x2x8x4xf32> to memref<8x4xf32, strided<[4, 1], offset: ?>>
      linalg.batch_reduce_matmul ins(%subview, %subview_0 : memref<4x8x2xf32, strided<[16, 2, 1], offset: ?>>, memref<4x2x4xf32, strided<[8, 4, 1], offset: ?>>) outs(%subview_1 : memref<8x4xf32, strided<[4, 1], offset: ?>>)
    }
    return
  }
}

// CHECK: func.func @entry(%[[ARG0:.*]]: memref<2x4x8x2xf32>, %[[ARG1:.*]]: memref<2x4x2x4xf32>, %[[ARG2:.*]]: memref<2x2x8x4xf32>) {
// CHECK: scf.forall (%[[ARG3:.*]], %[[ARG4:.*]]) in (2, 2) {
// CHECK:       %[[subview:.*]] = memref.subview %[[ARG0]][%[[ARG3]], 0, 0, 0] [1, 4, 8, 2] [1, 1, 1, 1] : memref<2x4x8x2xf32> to memref<4x8x2xf32, strided<[16, 2, 1], offset: ?>>
// CHECK:       %[[subview_0:.*]] = memref.subview %[[ARG1]][%[[ARG4]], 0, 0, 0] [1, 4, 2, 4] [1, 1, 1, 1] : memref<2x4x2x4xf32> to memref<4x2x4xf32, strided<[8, 4, 1], offset: ?>>
// CHECK:       %[[subview_1:.*]] = memref.subview %[[ARG2]][%[[ARG3]], %[[ARG4]], 0, 0] [1, 1, 8, 4] [1, 1, 1, 1] : memref<2x2x8x4xf32> to memref<8x4xf32, strided<[4, 1], offset: ?>>
// CHECK:       %[[vec0:.*]] = vector.transfer_read %[[subview]][%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<4x8x2xf32, strided<[16, 2, 1], offset: ?>>, vector<4x8x2xf32>
// CHECK:       %[[vec1:.*]] = vector.transfer_read %[[subview_0]][%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<4x2x4xf32, strided<[8, 4, 1], offset: ?>>, vector<4x2x4xf32>
// CHECK:       %[[vec2:.*]] = vector.transfer_read %[[subview_1]][%c0, %c0], %cst {in_bounds = [true, true]} : memref<8x4xf32, strided<[4, 1], offset: ?>>, vector<8x4xf32>
// CHECK:       %[[vec3:.*]] = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %0, %1, %2 : vector<4x8x2xf32>, vector<4x2x4xf32> into vector<8x4xf32>
// CHECK:       vector.transfer_write %[[vec3]], %[[subview_1]][%c0, %c0] {in_bounds = [true, true]} : vector<8x4xf32>, memref<8x4xf32, strided<[4, 1], offset: ?>>
// CHECK:     }

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4 floordiv 2, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
module {
  memref.global "private" constant @__constant_4x1x4x2xbf16 : memref<4x1x4x2xbf16> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @entry(%arg0: memref<2x4x8x2xbf16>) -> memref<2x2x8x4xbf16> {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = memref.get_global @__constant_4x1x4x2xbf16 : memref<4x1x4x2xbf16>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x2x8x4xbf16>
    scf.forall (%arg1, %arg2) in (2, 2) {
      %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 8, 4] [1, 1, 1, 1] : memref<2x2x8x4xbf16> to memref<8x4xbf16, strided<[4, 1], offset: ?>>
      linalg.fill ins(%cst : bf16) outs(%subview : memref<8x4xbf16, strided<[4, 1], offset: ?>>)
      %subview_0 = memref.subview %arg0[%arg1, 0, 0, 0] [1, 4, 8, 2] [1, 1, 1, 1] : memref<2x4x8x2xbf16> to memref<4x8x2xbf16, strided<[16, 2, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%subview_0, %0 : memref<4x8x2xbf16, strided<[16, 2, 1], offset: ?>>, memref<4x1x4x2xbf16>) outs(%subview : memref<8x4xbf16, strided<[4, 1], offset: ?>>) {
      ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
        %1 = arith.mulf %in, %in_1 : bf16
        %2 = arith.addf %out, %1 : bf16
        linalg.yield %2 : bf16
      }
    }
    return %alloc : memref<2x2x8x4xbf16>
  }
}

// CHECK:   func.func @entry(%[[ARG0:.*]]: memref<2x4x8x2xbf16>) -> memref<2x2x8x4xbf16> {
// CHECK:     scf.forall (%[[ARG1:.*]], %[[ARG2:.*]]) in (2, 2) {
// CHECK:       %[[subview:.*]] = memref.subview %alloc[%[[ARG1]], %[[ARG2]], 0, 0] [1, 1, 8, 4] [1, 1, 1, 1] : memref<2x2x8x4xbf16> to memref<8x4xbf16, strided<[4, 1], offset: ?>>
// CHECK:       vector.transfer_write %cst_0, %[[subview]][%c0, %c0] {in_bounds = [true, true]} : vector<8x4xbf16>, memref<8x4xbf16, strided<[4, 1], offset: ?>>
// CHECK:       %[[subview_1:.*]] = memref.subview %[[ARG0]][%[[ARG1]], 0, 0, 0] [1, 4, 8, 2] [1, 1, 1, 1] : memref<2x4x8x2xbf16> to memref<4x8x2xbf16, strided<[16, 2, 1], offset: ?>>
// CHECK:       %[[expand_shape:.*]] = memref.expand_shape %[[subview_1]] {{\[}}[0], [1], [2, 3]] output_shape [4, 8, 1, 2] : memref<4x8x2xbf16, strided<[16, 2, 1], offset: ?>> into memref<4x8x1x2xbf16, strided<[16, 2, 2, 1], offset: ?>>
// CHECK:       %[[vec1:.*]] = vector.transfer_read %[[expand_shape]][%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<4x8x1x2xbf16, strided<[16, 2, 2, 1], offset: ?>>, vector<4x8x1x2xbf16>
// CHECK:       %[[vec2:.*]] = vector.transfer_read %0[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<4x1x4x2xbf16>, vector<4x1x4x2xbf16>
// CHECK:       %[[vec3:.*]] = vector.transfer_read %[[subview]][%c0, %c0], %cst {in_bounds = [true, true]} : memref<8x4xbf16, strided<[4, 1], offset: ?>>, vector<8x4xbf16>
// CHECK:       %[[vec4:.*]] = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %2, %3 : vector<4x8x1x2xbf16>, vector<4x1x4x2xbf16> into vector<8x4xbf16>
// CHECK:       vector.transfer_write %[[vec4]], %[[subview]][%c0, %c0] {in_bounds = [true, true]} : vector<8x4xbf16>, memref<8x4xbf16, strided<[4, 1], offset: ?>>

