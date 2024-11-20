//RUN: tpp-opt --insert-transpose-pass --split-input-file %s | FileCheck %s

 memref.global "private" constant @__constant_5x2xbf16 : memref<5x2xbf16> = dense<[[0.000000e+00, 0.000000e+00], [1.748050e-01, 3.085940e-01], [3.144530e-01, 5.664060e-02], [6.542960e-02, 0.000000e+00], [1.457210e-03, 0.000000e+00]]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_5x5x2x2xbf16 : memref<5x5x2x2xbf16> = dense<[[[[0.000000e+00, 1.298830e-01], [1.513670e-01, 1.062010e-02]], [[3.757480e-04, 2.988280e-01], [9.814450e-02, 1.123050e-02]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 5.004880e-02]], [[1.289060e-01, 1.483150e-02], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]]], [[[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[1.562500e-01, 0.000000e+00], [0.000000e+00, 1.318360e-01]], [[2.070310e-01, 0.000000e+00], [6.494140e-02, 1.542970e-01]], [[1.865230e-01, 1.118160e-01], [3.886720e-01, 9.423820e-02]], [[1.884770e-01, 0.000000e+00], [1.445310e-01, 0.000000e+00]]], [[[0.000000e+00, 0.000000e+00], [0.000000e+00, 2.285160e-01]], [[0.000000e+00, 2.490230e-01], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 4.913330e-03], [2.539060e-01, 0.000000e+00]], [[0.000000e+00, 9.912100e-02], [0.000000e+00, 2.563480e-02]], [[0.000000e+00, 1.044920e-01], [0.000000e+00, 0.000000e+00]]], [[[0.000000e+00, 9.912100e-02], [2.421880e-01, 1.718750e-01]], [[0.000000e+00, 2.490230e-01], [2.465820e-02, 6.201170e-02]], [[0.000000e+00, 2.773440e-01], [0.000000e+00, 6.054690e-02]], [[0.000000e+00, 7.373050e-02], [2.285160e-01, 2.353520e-01]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 2.392580e-02]]], [[[1.239010e-02, 3.984380e-01], [2.233890e-02, 0.000000e+00]], [[9.619140e-02, 0.000000e+00], [0.000000e+00, 1.201170e-01]], [[0.000000e+00, 3.613280e-02], [0.000000e+00, 2.226560e-01]], [[0.000000e+00, 2.349850e-03], [6.079100e-02, 0.000000e+00]], [[4.394530e-02, 2.216800e-01], [0.000000e+00, 9.326170e-02]]]]> {alignment = 64 : i64}

func.func @entry(%arg0: memref<5x5x2x2xbf16>) -> memref<5x5x2x2xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %cst_0 = arith.constant dense<0.000000e+00> : vector<2x2xbf16>
  %c0 = arith.constant 0 : index
  %0 = memref.get_global @__constant_5x5x2x2xbf16 : memref<5x5x2x2xbf16>
  %1 = memref.get_global @__constant_5x2xbf16 : memref<5x2xbf16>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<5x5x2x2xbf16>
  scf.forall (%arg1, %arg2) in (5, 5) {
    %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 2, 2] [1, 1, 1, 1] : memref<5x5x2x2xbf16> to memref<2x2xbf16, strided<[2, 1], offset: ?>>
    vector.transfer_write %cst_0, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<2x2xbf16>, memref<2x2xbf16, strided<[2, 1], offset: ?>>
    %subview_1 = memref.subview %arg0[%arg1, 0, 0, 0] [1, 5, 2, 2] [1, 1, 1, 1] : memref<5x5x2x2xbf16> to memref<5x2x2xbf16, strided<[4, 2, 1], offset: ?>>
    %subview_2 = memref.subview %0[%arg2, 0, 0, 0] [1, 5, 2, 2] [1, 1, 1, 1] : memref<5x5x2x2xbf16> to memref<5x2x2xbf16, strided<[4, 2, 1], offset: ?>>
    %2 = vector.transfer_read %subview_1[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<5x2x2xbf16, strided<[4, 2, 1], offset: ?>>, vector<5x2x2xbf16>
    %3 = vector.transfer_read %subview_2[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<5x2x2xbf16, strided<[4, 2, 1], offset: ?>>, vector<5x2x2xbf16>
    %4 = vector.transfer_read %subview[%c0, %c0], %cst {in_bounds = [true, true]} : memref<2x2xbf16, strided<[2, 1], offset: ?>>, vector<2x2xbf16>
    %5 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %2, %3, %4 : vector<5x2x2xbf16>, vector<5x2x2xbf16> into vector<2x2xbf16>
    vector.transfer_write %5, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<2x2xbf16>, memref<2x2xbf16, strided<[2, 1], offset: ?>>
    %subview_3 = memref.subview %1[%arg2, 0] [1, 2] [1, 1] : memref<5x2xbf16> to memref<2xbf16, strided<[1], offset: ?>>
    %6 = vector.transfer_read %subview_3[%c0], %cst {in_bounds = [true]} : memref<2xbf16, strided<[1], offset: ?>>, vector<2xbf16>
    %7 = vector.broadcast %6 : vector<2xbf16> to vector<2x2xbf16>
    %8 = vector.transfer_read %subview[%c0, %c0], %cst {in_bounds = [true, true]} : memref<2x2xbf16, strided<[2, 1], offset: ?>>, vector<2x2xbf16>
    %9 = arith.addf %7, %8 : vector<2x2xbf16>
    vector.transfer_write %9, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<2x2xbf16>, memref<2x2xbf16, strided<[2, 1], offset: ?>>
    %10 = vector.transfer_read %subview[%c0, %c0], %cst {in_bounds = [true, true]} : memref<2x2xbf16, strided<[2, 1], offset: ?>>, vector<2x2xbf16>
    %11 = arith.maximumf %10, %cst_0 : vector<2x2xbf16>
    vector.transfer_write %11, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<2x2xbf16>, memref<2x2xbf16, strided<[2, 1], offset: ?>>
  }
  return %alloc : memref<5x5x2x2xbf16>
}

// CHECK-LABEL: func.func @entry(
// CHECK: %[[ARG0:.*]]: memref<5x5x2x2xbf16>) -> memref<5x5x2x2xbf16> {
// CHECK-DAG:   %[[cst:.*]] = arith.constant 0.000000e+00 : bf16
// CHECK-DAG:  %[[cst_0:.*]] = arith.constant dense<0.000000e+00>
// CHECK-DAG:  %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[global0:.*]] = memref.get_global @__constant_5x5x2x2xbf16
// CHECK-DAG:  %[[global1:.*]] = memref.get_global @__constant_5x2xbf16
// CHECK:  %[[alloc:.*]] = memref.alloc() {alignment = 64 : i64}
// CHECK:  scf.forall (%[[ARG1:.*]], %[[ARG2:.*]]) in (5, 5) {
// CHECK:    %[[subview:.*]] = memref.subview %alloc[%[[ARG1]], %[[ARG2]], 0, 0] [1, 1, 2, 2] [1, 1, 1, 1]
// CHECK:    vector.transfer_write %[[cst_0]], %[[subview]][%[[c0]], %[[c0]]] {in_bounds = [true, true]}
// CHECK-DAG: %[[subview_1:.*]] = memref.subview %[[ARG0]][%[[ARG1]], 0, 0, 0] [1, 5, 2, 2] [1, 1, 1, 1]
// CHECK-DAG: %[[subview_2:.*]] = memref.subview %[[global0]][%[[ARG2]], 0, 0, 0] [1, 5, 2, 2] [1, 1, 1, 1]
// CHECK-DAG: %[[read2:.*]] = vector.transfer_read %[[subview_1]][%[[c0]], %[[c0]], %[[c0]]], %[[cst]] {in_bounds = [true, true, true]}
// CHECK-DAG: %[[read3:.*]] = vector.transfer_read %[[subview_2]][%[[c0]], %[[c0]], %[[c0]]], %cst {in_bounds = [true, true, true]}
// CHECK-DAG: %[[read4:.*]] = vector.transfer_read %[[subview]][%[[c0]], %[[c0]]], %[[cst]] {in_bounds = [true, true]}
// CHECK:     %[[read5:.*]] = vector.transpose %[[read3]], [0, 2, 1]
// CHECK:    %[[alloca:.*]] = memref.alloca()
// CHECK:    vector.transfer_write %[[read5]], %[[alloca]][%[[c0]], %[[c0]], %[[c0]]] {in_bounds = [true, true, true]}
// CHECK:    %[[read6:.*]] = vector.transfer_read %alloca[%[[c0]], %[[c0]], %[[c0]]], %cst {in_bounds = [true, true, true]}
// CHECK:    %[[contract:.*]] = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[read2]], %[[read6]], %[[read4]]
// CHECK:    vector.transfer_write %[[contract]], %[[subview]][%[[c0]], %[[c0]]] {in_bounds = [true, true]}
// CHECK:    %[[subview_3:.*]] = memref.subview %[[global1]][%[[ARG2]], 0] [1, 2] [1, 1]
// CHECK:    %[[read6:.*]] = vector.transfer_read %[[subview_3]][%[[c0]]], %[[cst]] {in_bounds = [true]}
// CHECK:    %[[read7:.*]] = vector.broadcast %[[read6]]
// CHECK:    %[[read8:.*]] = vector.transfer_read %[[subview]][%[[c0]], %[[c0]]], %[[cst]] {in_bounds = [true, true]}
// CHECK:    %[[read9:.*]] = arith.addf %[[read7]], %[[read8]]
// CHECK:    vector.transfer_write %[[read9]], %[[subview]][%[[c0]], %[[c0]]] {in_bounds = [true, true]}
// CHECK:    %[[read10:.*]] = vector.transfer_read %[[subview]][%[[c0]], %[[c0]]], %[[cst]] {in_bounds = [true, true]}
// CHECK:    %[[read11:.*]] = arith.maximumf %[[read10]], %[[cst_0]]
// CHECK:    vector.transfer_write %[[read11]], %[[subview]][%[[c0]], %[[c0]]] {in_bounds = [true, true]}

// -----

func.func @matmul_static(%arg0: memref<4x8xf32>, %arg1: memref<16x8xf32>, %arg2: memref<4x16xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant dense<0.000000e+00> : vector<8x2xf32>
  %cst_1 = arith.constant -1.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %expand_shape = memref.expand_shape %arg0 [[0, 1], [2, 3]] output_shape [2, 2, 2, 4] : memref<4x8xf32> into memref<2x2x2x4xf32>
  %expand_shape_2 = memref.expand_shape %arg1 [[0, 1], [2, 3]] output_shape [2, 8, 2, 4] : memref<16x8xf32> into memref<2x8x2x4xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x2x8x2xf32>
  scf.forall (%arg3, %arg4) in (2, 2) {
    %subview = memref.subview %alloc[%arg3, %arg4, 0, 0] [1, 1, 8, 2] [1, 1, 1, 1] : memref<2x2x8x2xf32> to memref<8x2xf32, strided<[2, 1], offset: ?>>
    vector.transfer_write %cst_0, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<8x2xf32>, memref<8x2xf32, strided<[2, 1], offset: ?>>
    %subview_3 = memref.subview %expand_shape[%arg3, 0, %arg4, 0] [1, 2, 1, 4] [1, 1, 1, 1] : memref<2x2x2x4xf32> to memref<2x4xf32, strided<[8, 1], offset: ?>>
    %subview_4 = memref.subview %expand_shape_2[%arg3, 0, %arg4, 0] [1, 8, 1, 4] [1, 1, 1, 1] : memref<2x8x2x4xf32> to memref<8x4xf32, strided<[8, 1], offset: ?>>
    %1 = vector.transfer_read %subview_3[%c0, %c0], %cst {in_bounds = [true, true]} : memref<2x4xf32, strided<[8, 1], offset: ?>>, vector<2x4xf32>
    %2 = vector.transfer_read %subview_4[%c0, %c0], %cst {in_bounds = [true, true]} : memref<8x4xf32, strided<[8, 1], offset: ?>>, vector<8x4xf32>
    %3 = vector.transfer_read %subview[%c0, %c0], %cst {in_bounds = [true, true]} : memref<8x2xf32, strided<[2, 1], offset: ?>>, vector<8x2xf32>
    %4 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d2, d0)>], iterator_types = ["parallel", "reduction", "parallel"], kind = #vector.kind<add>} %1, %2, %3 : vector<2x4xf32>, vector<8x4xf32> into vector<8x2xf32>
    vector.transfer_write %4, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<8x2xf32>, memref<8x2xf32, strided<[2, 1], offset: ?>>
  }
  %collapse_shape = memref.collapse_shape %alloc [[0, 1], [2, 3]] : memref<2x2x8x2xf32> into memref<4x16xf32>
  %0 = vector.transfer_read %collapse_shape[%c0, %c0], %cst_1 {in_bounds = [true, true]} : memref<4x16xf32>, vector<4x16xf32>
  vector.print %0 : vector<4x16xf32>
  memref.dealloc %alloc : memref<2x2x8x2xf32>
  return
}

// CHECK-LABEL:  func.func @matmul_static(
// CHECK: %[[arg0:.*]]: memref<4x8xf32>, %[[arg1:.*]]: memref<16x8xf32>, %[[arg2:.*]]: memref<4x16xf32>) {
// CHECK:         %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[cst:.*]] = arith.constant -1.000000e+00 : f32
// CHECK-DAG:     %[[cst_0:.*]] = arith.constant dense<0.000000e+00> : vector<8x2xf32>
// CHECK-DAG:     %[[cst_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %[[c29_i64:.*]] = arith.constant 29 : i64
// CHECK-DAG:     %[[c1_i64:.*]] = arith.constant 1 : i64
// CHECK-DAG:     %[[c2_i64:.*]] = arith.constant 2 : i64
// CHECK-DAG:     %[[c4_i64:.*]] = arith.constant 4 : i64
// CHECK-DAG:     %[[c0_i64:.*]] = arith.constant 0 : i64
// CHECK-DAG:     %[[dispatch:.*]] = call @xsmm_unary_dispatch(%[[c29_i64]], %[[c1_i64]], %[[c2_i64]], %[[c4_i64]], %[[c4_i64]], %[[c2_i64]], %[[c0_i64]])
// CHECK-DAG:     %[[expand_shape:.*]] = memref.expand_shape %[[arg0]] {{\[}}[0, 1], [2, 3]] output_shape [2, 2, 2, 4]
// CHECK-DAG:     %[[expand_shape_2:.*]] = memref.expand_shape %arg1 {{\[}}[0, 1], [2, 3]] output_shape [2, 8, 2, 4]
// CHECK-DAG:     %[[alloc:.*]] = memref.alloc() {alignment = 64 : i64}
// CHECK:         scf.forall (%[[arg3:.*]], %[[arg4:.*]]) in (2, 2) {
// CHECK:       %[[subview:.*]] = memref.subview %[[alloc]][%[[arg3]], %[[arg4]], 0, 0] [1, 1, 8, 2] [1, 1, 1, 1]
// CHECK:       vector.transfer_write %[[cst_0]], %[[subview]][%[[c0]], %[[c0]]] {in_bounds = [true, true]}
// CHECK:       %[[subview_3:.*]] = memref.subview %[[expand_shape]][%[[arg3]], 0, %[[arg4]], 0] [1, 2, 1, 4] [1, 1, 1, 1]
// CHECK:       %[[subview_4:.*]] = memref.subview %[[expand_shape_2]][%[[arg3]], 0, %[[arg4]], 0] [1, 8, 1, 4] [1, 1, 1, 1]
// CHECK:       %[[read2:.*]] = vector.transfer_read %[[subview_4]][%[[c0]], %[[c0]]], %[[cst_1]] {in_bounds = [true, true]}
// CHECK:       %[[read3:.*]] = vector.transfer_read %[[subview]][%[[c0]], %[[c0]]], %[[cst_1]] {in_bounds = [true, true]}
// CHECK:       %[[alloca:.*]] = memref.alloca()
// CHECK:       %[[base_buffer:.*]], %[[offset:.*]], %[[sizes:.*]]:2, %[[strides:.*]]:2 = memref.extract_strided_metadata %[[subview_3]]
// CHECK:       %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[subview_3]]
// CHECK:       %[[indexcast4:.*]] = arith.index_cast %[[intptr]]
// CHECK:       %[[inttoptr5:.*]] = llvm.inttoptr %[[indexcast4]]
// CHECK:       %[[intptr5:.*]] = memref.extract_aligned_pointer_as_index %[[alloca]]
// CHECK:       %[[indexcast6:.*]] = arith.index_cast %[[intptr5]]
// CHECK:       %[[inttoptr7:.*]] = llvm.inttoptr %6 : i64 to !llvm.ptr
// CHECK:       func.call @xsmm_unary_invoke(%[[c1_i64]], %[[dispatch]], %[[inttoptr5]], %[[offset]], %[[inttoptr7]], %[[c0]])
// CHECK:       %[[read8:.*]] = vector.transfer_read %[[alloca]][%[[c0]], %[[c0]]], %[[cst_1]] {in_bounds = [true, true]}
// CHECK:       %[[read9:.*]] = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction", "parallel"], kind = #vector.kind<add>} %[[read2]], %[[read8]], %[[read3]]
// CHECK:       vector.transfer_write %[[read9]], %[[subview]][%[[c0]], %[[c0]]] {in_bounds = [true, true]}

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d2)>
module {
  func.func @vnni_brgemm_require_transpose_on_C(%arg0: memref<16x32x32xbf16>, %arg1: memref<16x16x32x2xbf16>, %arg2: memref<32x32xbf16>) {
    %cst = arith.constant 0.000000e+00 : bf16
    %c0 = arith.constant 0 : index
    %expand_shape = memref.expand_shape %arg0 [[0], [1], [2, 3]] output_shape [16, 32, 16, 2] : memref<16x32x32xbf16> into memref<16x32x16x2xbf16>
    %0 = vector.transfer_read %expand_shape[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<16x32x16x2xbf16>, vector<16x32x16x2xbf16>
    %1 = vector.transfer_read %arg1[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<16x16x32x2xbf16>, vector<16x16x32x2xbf16>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<32x32xbf16>, vector<32x32xbf16>
    %3 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %0, %1, %2 : vector<16x32x16x2xbf16>, vector<16x16x32x2xbf16> into vector<32x32xbf16>
    vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xbf16>, memref<32x32xbf16>
    return
  }
}

// CHECK: #map = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4, d0, d1)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d2)>
// CHECK: module {
// CHECK-LABEL:  func.func @vnni_brgemm_require_transpose_on_C(
// CHECK: %[[arg0:.*]]: memref<16x32x32xbf16>, %[[arg1:.*]]: memref<16x16x32x2xbf16>, %[[arg2:.*]]: memref<32x32xbf16>) {
// CHECK-DAG:  %[[cst:.*]] = arith.constant 0.000000e+00 : bf16
// CHECK-DAG:  %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[expand_shape:.*]] = memref.expand_shape %[[arg0]] {{\[}}[0], [1], [2, 3]] output_shape [16, 32, 16, 2]
// CHECK-DAG:  %[[read0:.*]] = vector.transfer_read %[[expand_shape]][%[[c0]], %[[c0]], %[[c0]], %[[c0]]], %[[cst]] {in_bounds = [true, true, true, true]}
// CHECK-DAG:  %[[read1:.*]] = vector.transfer_read %[[arg1]][%[[c0]], %[[c0]], %[[c0]], %[[c0]]], %[[cst]] {in_bounds = [true, true, true, true]}
// CHECK-DAG:  %[[read2:.*]] = vector.transfer_read %[[arg2]][%[[c0]], %[[c0]]], %[[cst]] {in_bounds = [true, true]}
// CHECK:      %[[transpose:.*]] = vector.transpose %[[read1]], [2, 1, 0, 3] : vector<16x16x32x2xbf16> to vector<32x16x16x2xbf16>
// CHECK:      %[[alloca:.*]] = memref.alloca() : memref<32x16x16x2xbf16>
// CHECK:      vector.transfer_write %[[read3]], %alloca[%[[c0]], %[[c0]], %[[c0]], %[[c0]]] {in_bounds = [true, true, true, true]}
// CHECK:      %[[read4:.*]] = vector.transfer_read %[[alloca]][%[[c0]], %[[c0]], %[[c0]], %[[c0]]], %[[cst]] {in_bounds = [true, true, true, true]}
// CHECK:      %[[read5:.*]] = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[read4]], %[[read0]], %[[read2]]
// CHECK:    vector.transfer_write %[[read5]], %[[arg2]][%[[c0]], %[[c0]]] {in_bounds = [true, true]}
