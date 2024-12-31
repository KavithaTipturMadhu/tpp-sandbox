// RUN: tpp-opt --vector-to-xsmm  %s --split-input-file | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d3)>
module {
  func.func @brgemm(%arg0: memref<2x2x2x4xf32>, %arg1: memref<2x4x8x2xf32>, %arg2: memref<2x2x8x2xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    scf.forall (%arg3, %arg4) in (2, 8) {
      %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 2, 2, 4] [1, 1, 1, 1] : memref<2x2x2x4xf32> to memref<2x2x4xf32, strided<[8, 4, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[0, 0, %arg4, 0] [2, 4, 1, 2] [1, 1, 1, 1] : memref<2x4x8x2xf32> to memref<2x4x2xf32, strided<[64, 16, 1], offset: ?>>
      %subview_1 = memref.subview %arg2[%arg3, 0, %arg4, 0] [1, 2, 1, 2] [1, 1, 1, 1] : memref<2x2x8x2xf32> to memref<2x2xf32, strided<[16, 1], offset: ?>>
      %0 = vector.transfer_read %subview[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<2x2x4xf32, strided<[8, 4, 1], offset: ?>>, vector<2x2x4xf32>
      %1 = vector.transfer_read %subview_0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<2x4x2xf32, strided<[64, 16, 1], offset: ?>>, vector<2x4x2xf32>
      %2 = vector.transfer_read %subview_1[%c0, %c0], %cst {in_bounds = [true, true]} : memref<2x2xf32, strided<[16, 1], offset: ?>>, vector<2x2xf32>
      %3 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction", "reduction", "parallel"], kind = #vector.kind<add>} %0, %1, %2 : vector<2x2x4xf32>, vector<2x4x2xf32> into vector<2x2xf32>
      vector.transfer_write %3, %subview_1[%c0, %c0] {in_bounds = [true, true]} : vector<2x2xf32>, memref<2x2xf32, strided<[16, 1], offset: ?>>
    }
    return
  }
}
// CHECK-LABEL:  func.func @brgemm(
// CHECK: %[[arg0:.*]]: memref<2x2x2x4xf32>, %[[arg1:.*]]: memref<2x4x8x2xf32>, %[[arg2:.*]]: memref<2x2x8x2xf32>) {
// CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[c2_i64:.*]] = arith.constant 2 : i64
// CHECK-DAG: %[[c4_i64:.*]] = arith.constant 4 : i64
// CHECK-DAG: %[[c8_i64:.*]] = arith.constant 8 : i64
// CHECK-DAG: %[[c16_i64:.*]] = arith.constant 16 : i64
// CHECK-DAG: %[[c64_i64:.*]] = arith.constant 64 : i64
// CHECK-DAG: %[[c0_i64:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[dispatch:.*]] = call @xsmm_brgemm_dispatch(%[[c1_i64]], %[[c2_i64]], %[[c2_i64]], %[[c4_i64]], %[[c8_i64]], %[[c16_i64]], %[[c16_i64]], %[[c4_i64]], %[[c64_i64]], %[[c0_i64]])
// CHECK: scf.forall (%[[arg3:.*]], %[[arg4:.*]]) in (2, 8) {
// CHECK-DAG:      %[[subview:.*]] = memref.subview %[[arg0]][%[[arg3]], 0, 0, 0] [1, 2, 2, 4] [1, 1, 1, 1]
// CHECK-DAG:      %[[subview_0:.*]] = memref.subview %[[arg1]][0, 0, %[[arg4]], 0] [2, 4, 1, 2] [1, 1, 1, 1]
// CHECK-DAG:      %[[subview_1:.*]] = memref.subview %[[arg2]][%[[arg3]], 0, %[[arg4]], 0] [1, 2, 1, 2] [1, 1, 1, 1]
// CHECK-DAG:      %[[base_buffer:.*]], %[[offset:.*]], %[[sizes:.*]]:3, %[[strides:.*]]:3 = memref.extract_strided_metadata %[[subview]]
// CHECK:          %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[subview]]
// CHECK-NEXT:     %[[cast:.*]] = arith.index_cast %intptr : index to i64
// CHECK-NEXT:     %[[inttoptr:.*]] = llvm.inttoptr %[[cast]]
// CHECK-DAG:      %[[base_buffer_2:.*]], %[[offset_3:.*]], %[[sizes_4:.*]]:3, %[[strides_5:.*]]:3 = memref.extract_strided_metadata %[[subview_0]] 
// CHECK-DAG:      %[[intptr_6:.*]] = memref.extract_aligned_pointer_as_index %[[subview_0]]
// CHECK-NEXT:     %[[cast2:.*]] = arith.index_cast %[[intptr_6]]
// CHECK-NEXT:     %[[inttoptr2:.*]] = llvm.inttoptr %[[cast2]]
// CHECK-DAG:      %[[base_buffer_7:.*]], %[[offset_8:.*]], %[[sizes_9:.*]]:2, %[[strides_10:.*]]:2 = memref.extract_strided_metadata %[[subview_1]]
// CHECK-DAG:      %[[intptr_11:.*]] = memref.extract_aligned_pointer_as_index %[[subview_1]]
// CHECK-NEXT:     %[[cast3:.*]] = arith.index_cast %[[intptr_11]]
// CHECK-NEXT:     %[[inttoptr3:.*]] = llvm.inttoptr %[[cast3]]
// CHECK:          func.call @xsmm_brgemm_invoke(%[[c1_i64]], %[[dispatch]], %[[inttoptr]], %[[offset]], %[[inttoptr2]], %[[offset_3]], %[[inttoptr3]], %[[offset_8]], %[[c2_i64]])

// -----
#map = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
module {
  func.func @brgemm_1(%arg0: memref<9x4x5xf32>, %arg1: memref<9x5x8xf32>, %arg2: memref<4x8xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<9x4x5xf32>, vector<9x4x5xf32>
    %1 = vector.transfer_read %arg1[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<9x5x8xf32>, vector<9x5x8xf32>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x8xf32>, vector<4x8xf32>
    %3 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction"], kind = #vector.kind<add>} %0, %1, %2 : vector<9x4x5xf32>, vector<9x5x8xf32> into vector<4x8xf32>
    vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<4x8xf32>, memref<4x8xf32>
    return
  }
}
// CHECK-LABEL:  func.func @brgemm_1(
// CHECK: %[[arg0:.*]]: memref<9x4x5xf32>, %[[arg1:.*]]: memref<9x5x8xf32>, %[[arg2:.*]]: memref<4x8xf32>) {
// CHECK-DAG: %[[c9_i64:.*]] = arith.constant 9 : i64
// CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[c4_i64:.*]] = arith.constant 4 : i64
// CHECK-DAG: %[[c8_i64:.*]] = arith.constant 8 : i64
// CHECK-DAG: %[[c5_i64:.*]] = arith.constant 5 : i64
// CHECK-DAG: %[[c20_i64:.*]] = arith.constant 20 : i64
// CHECK-DAG: %[[c40_i64:.*]] = arith.constant 40 : i64
// CHECK-DAG: %[[c0_i64:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[dispatch:.*]] = call @xsmm_brgemm_dispatch(%[[c1_i64]], %[[c4_i64]], %[[c8_i64]], %[[c5_i64]], %[[c5_i64]], %[[c8_i64]], %[[c8_i64]], %[[c20_i64]], %[[c40_i64]], %[[c0_i64]])
// CHECK:     %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[arg0]]
// CHECK-NEXT:%[[cast:.*]] = arith.index_cast %intptr : index to i64
// CHECK-NEXT:%[[inttoptr:.*]] = llvm.inttoptr %[[cast]]
// CHECK-DAG: %[[intptr_6:.*]] = memref.extract_aligned_pointer_as_index %[[arg1]]
// CHECK-NEXT:%[[cast2:.*]] = arith.index_cast %[[intptr_6]]
// CHECK-NEXT:%[[inttoptr2:.*]] = llvm.inttoptr %[[cast2]]
// CHECK-DAG: %[[intptr_11:.*]] = memref.extract_aligned_pointer_as_index %[[arg2]]
// CHECK-NEXT:%[[cast3:.*]] = arith.index_cast %[[intptr_11]]
// CHECK-NEXT:%[[inttoptr3:.*]] = llvm.inttoptr %[[cast3]]
// CHECK:     func.call @xsmm_brgemm_invoke(%[[c1_i64]], %[[dispatch]], %[[inttoptr]], %[[c0]], %[[inttoptr2]], %[[c0]], %[[inttoptr3]], %[[c0]], %[[c9_i64]])

// -----
#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>

// Changed iterator types order as compared to the previous test
module {
  func.func @brgemm_2(%arg0: memref<9x4x5xf32>, %arg1: memref<9x5x8xf32>, %arg2: memref<4x8xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<9x4x5xf32>, vector<9x4x5xf32>
    %1 = vector.transfer_read %arg1[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<9x5x8xf32>, vector<9x5x8xf32>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x8xf32>, vector<4x8xf32>
    %3 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel"], kind = #vector.kind<add>} %0, %1, %2 : vector<9x4x5xf32>, vector<9x5x8xf32> into vector<4x8xf32>
    vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<4x8xf32>, memref<4x8xf32>
    return
  }
}

// CHECK-LABEL:  func.func @brgemm_2(
// CHECK: %[[arg0:.*]]: memref<9x4x5xf32>, %[[arg1:.*]]: memref<9x5x8xf32>, %[[arg2:.*]]: memref<4x8xf32>) {
// CHECK-DAG: %[[c9_i64:.*]] = arith.constant 9 : i64
// CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[c4_i64:.*]] = arith.constant 4 : i64
// CHECK-DAG: %[[c8_i64:.*]] = arith.constant 8 : i64
// CHECK-DAG: %[[c5_i64:.*]] = arith.constant 5 : i64
// CHECK-DAG: %[[c20_i64:.*]] = arith.constant 20 : i64
// CHECK-DAG: %[[c40_i64:.*]] = arith.constant 40 : i64
// CHECK-DAG: %[[c0_i64:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[dispatch:.*]] = call @xsmm_brgemm_dispatch(%[[c1_i64]], %[[c4_i64]], %[[c8_i64]], %[[c5_i64]], %[[c5_i64]], %[[c8_i64]], %[[c8_i64]], %[[c20_i64]], %[[c40_i64]], %[[c0_i64]])
// CHECK:     %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[arg0]]
// CHECK-NEXT:%[[cast:.*]] = arith.index_cast %intptr : index to i64
// CHECK-NEXT:%[[inttoptr:.*]] = llvm.inttoptr %[[cast]]
// CHECK-DAG: %[[intptr_6:.*]] = memref.extract_aligned_pointer_as_index %[[arg1]]
// CHECK-NEXT:%[[cast2:.*]] = arith.index_cast %[[intptr_6]]
// CHECK-NEXT:%[[inttoptr2:.*]] = llvm.inttoptr %[[cast2]]
// CHECK-DAG: %[[intptr_11:.*]] = memref.extract_aligned_pointer_as_index %[[arg2]]
// CHECK-NEXT:%[[cast3:.*]] = arith.index_cast %[[intptr_11]]
// CHECK-NEXT:%[[inttoptr3:.*]] = llvm.inttoptr %[[cast3]]
// CHECK:     func.call @xsmm_brgemm_invoke(%[[c1_i64]], %[[dispatch]], %[[inttoptr]], %[[c0]], %[[inttoptr2]], %[[c0]], %[[inttoptr3]], %[[c0]], %[[c9_i64]])

// -----
#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
module {
  func.func @brgemm_invalid_strided_inputs(%arg0: memref<9x4x5xf32>, %arg1: memref<9x5x8xf32, strided<[40, 8, 2], offset: ?>>, %arg2: memref<4x8xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<9x4x5xf32>, vector<9x4x5xf32>
    %1 = vector.transfer_read %arg1[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<9x5x8xf32, strided<[40, 8, 2], offset: ?>>, vector<9x5x8xf32>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x8xf32>, vector<4x8xf32>
    %3 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel"], kind = #vector.kind<add>} %0, %1, %2 : vector<9x4x5xf32>, vector<9x5x8xf32> into vector<4x8xf32>
    vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<4x8xf32>, memref<4x8xf32>
    return
  }
}
// CHECK-LABEL:  func.func @brgemm_2(
// CHECK: %[[arg0:.*]]: memref<9x4x5xf32>, %[[arg1:.*]]: memref<9x5x8xf32, strided<[40, 8, 2], offset: ?>>, %[[arg2:.*]]: memref<4x8xf32>) {
// CHECK-NOT: call @xsmm_brgemm_dispatch
// CHECK-NOT: func.call @xsmm_brgemm_invoke

// -----
#map = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
module {
  func.func @brgemm_5(%arg0: memref<9x4x5xf32>, %arg1: memref<9x8x5xf32>, %arg2: memref<4x8xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<9x4x5xf32>, vector<9x4x5xf32>
    %1 = vector.transfer_read %arg1[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<9x8x5xf32>, vector<9x8x5xf32>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x8xf32>, vector<4x8xf32>
    %3 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction"], kind = #vector.kind<add>} %0, %1, %2 : vector<9x4x5xf32>, vector<9x8x5xf32> into vector<4x8xf32>
    vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<4x8xf32>, memref<4x8xf32>
    return
  }
}
// CHECK-LABEL:  func.func @brgemm_5(
// CHECK: %[[arg0:.*]]: memref<2x2x2x4xf32>, %[[arg1:.*]]: memref<2x4x8x2xf32>, %[[arg2:.*]]: memref<2x2x8x2xf32>) {
// CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[c2_i64:.*]] = arith.constant 2 : i64
// CHECK-DAG: %[[c4_i64:.*]] = arith.constant 4 : i64
// CHECK-DAG: %[[c8_i64:.*]] = arith.constant 8 : i64
// CHECK-DAG: %[[c16_i64:.*]] = arith.constant 16 : i64
// CHECK-DAG: %[[c64_i64:.*]] = arith.constant 64 : i64
// CHECK-DAG: %[[c0_i64:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[dispatch:.*]] = call @xsmm_brgemm_dispatch(%[[c1_i64]], %[[c2_i64]], %[[c2_i64]], %[[c4_i64]], %[[c8_i64]], %[[c16_i64]], %[[c16_i64]], %[[c4_i64]], %[[c64_i64]], %[[c0_i64]])
// CHECK: scf.forall (%[[arg3:.*]], %[[arg4:.*]]) in (2, 8) {
// CHECK-DAG:      %[[subview:.*]] = memref.subview %[[arg0]][%[[arg3]], 0, 0, 0] [1, 2, 2, 4] [1, 1, 1, 1]
// CHECK-DAG:      %[[subview_0:.*]] = memref.subview %[[arg1]][0, 0, %[[arg4]], 0] [2, 4, 1, 2] [1, 1, 1, 1]
// CHECK-DAG:      %[[subview_1:.*]] = memref.subview %[[arg2]][%[[arg3]], 0, %[[arg4]], 0] [1, 2, 1, 2] [1, 1, 1, 1]
// CHECK-DAG:      %[[base_buffer:.*]], %[[offset:.*]], %[[sizes:.*]]:3, %[[strides:.*]]:3 = memref.extract_strided_metadata %[[subview]]
// CHECK:          %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[subview]]
// CHECK-NEXT:     %[[cast:.*]] = arith.index_cast %intptr : index to i64
// CHECK-NEXT:     %[[inttoptr:.*]] = llvm.inttoptr %[[cast]]
// CHECK-DAG:      %[[base_buffer_2:.*]], %[[offset_3:.*]], %[[sizes_4:.*]]:3, %[[strides_5:.*]]:3 = memref.extract_strided_metadata %[[subview_0]]
// CHECK-DAG:      %[[intptr_6:.*]] = memref.extract_aligned_pointer_as_index %[[subview_0]]
// CHECK-NEXT:     %[[cast2:.*]] = arith.index_cast %[[intptr_6]]
// CHECK-NEXT:     %[[inttoptr2:.*]] = llvm.inttoptr %[[cast2]]
// CHECK-DAG:      %[[base_buffer_7:.*]], %[[offset_8:.*]], %[[sizes_9:.*]]:2, %[[strides_10:.*]]:2 = memref.extract_strided_metadata %[[subview_1]]
// CHECK-DAG:      %[[intptr_11:.*]] = memref.extract_aligned_pointer_as_index %[[subview_1]]
// CHECK-NEXT:     %[[cast3:.*]] = arith.index_cast %[[intptr_11]]
// CHECK-NEXT:     %[[inttoptr3:.*]] = llvm.inttoptr %[[cast3]]
// CHECK:          func.call @xsmm_brgemm_invoke(%[[c1_i64]], %[[0]], %[[2]], %[[offset]], %[[4]], %[[offset_3]], %[[6]], %[[offset_8]], %[[c2_i64]])

// -----
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @gemm_1(%arg0: memref<64x32xf32>, %arg1: memref<32x64xf32>, %arg2: memref<64x64xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<64x32xf32>, vector<64x32xf32>
    %1 = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : memref<32x64xf32>, vector<32x64xf32>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<64x64xf32>, vector<64x64xf32>
    %3 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %0, %1, %2 : vector<64x32xf32>, vector<32x64xf32> into vector<64x64xf32>
    vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<64x64xf32>, memref<64x64xf32>
    return
  }
}
// CHECK-LABEL:  func.func @gemm_1(
// CHECK: %[[arg0:.*]]: memref<2x2x2x4xf32>, %[[arg1:.*]]: memref<2x4x8x2xf32>, %[[arg2:.*]]: memref<2x2x8x2xf32>) {
// CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[c2_i64:.*]] = arith.constant 2 : i64
// CHECK-DAG: %[[c4_i64:.*]] = arith.constant 4 : i64
// CHECK-DAG: %[[c8_i64:.*]] = arith.constant 8 : i64
// CHECK-DAG: %[[c16_i64:.*]] = arith.constant 16 : i64
// CHECK-DAG: %[[c64_i64:.*]] = arith.constant 64 : i64
// CHECK-DAG: %[[c0_i64:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[dispatch:.*]] = call @xsmm_brgemm_dispatch(%[[c1_i64]], %[[c2_i64]], %[[c2_i64]], %[[c4_i64]], %[[c8_i64]], %[[c16_i64]], %[[c16_i64]], %[[c4_i64]], %[[c64_i64]], %[[c0_i64]])
// CHECK: scf.forall (%[[arg3:.*]], %[[arg4:.*]]) in (2, 8) {
// CHECK-DAG:      %[[subview:.*]] = memref.subview %[[arg0]][%[[arg3]], 0, 0, 0] [1, 2, 2, 4] [1, 1, 1, 1]
// CHECK-DAG:      %[[subview_0:.*]] = memref.subview %[[arg1]][0, 0, %[[arg4]], 0] [2, 4, 1, 2] [1, 1, 1, 1]
// CHECK-DAG:      %[[subview_1:.*]] = memref.subview %[[arg2]][%[[arg3]], 0, %[[arg4]], 0] [1, 2, 1, 2] [1, 1, 1, 1]
// CHECK-DAG:      %[[base_buffer:.*]], %[[offset:.*]], %[[sizes:.*]]:3, %[[strides:.*]]:3 = memref.extract_strided_metadata %[[subview]]
// CHECK:          %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[subview]]
// CHECK-NEXT:     %[[cast:.*]] = arith.index_cast %intptr : index to i64
// CHECK-NEXT:     %[[inttoptr:.*]] = llvm.inttoptr %[[cast]]
// CHECK-DAG:      %[[base_buffer_2:.*]], %[[offset_3:.*]], %[[sizes_4:.*]]:3, %[[strides_5:.*]]:3 = memref.extract_strided_metadata %[[subview_0]]
// CHECK-DAG:      %[[intptr_6:.*]] = memref.extract_aligned_pointer_as_index %[[subview_0]]
// CHECK-NEXT:     %[[cast2:.*]] = arith.index_cast %[[intptr_6]]
// CHECK-NEXT:     %[[inttoptr2:.*]] = llvm.inttoptr %[[cast2]]
// CHECK-DAG:      %[[base_buffer_7:.*]], %[[offset_8:.*]], %[[sizes_9:.*]]:2, %[[strides_10:.*]]:2 = memref.extract_strided_metadata %[[subview_1]]
// CHECK-DAG:      %[[intptr_11:.*]] = memref.extract_aligned_pointer_as_index %[[subview_1]]
// CHECK-NEXT:     %[[cast3:.*]] = arith.index_cast %[[intptr_11]]
// CHECK-NEXT:     %[[inttoptr3:.*]] = llvm.inttoptr %[[cast3]]
// CHECK:          func.call @xsmm_brgemm_invoke(%[[c1_i64]], %[[0]], %[[2]], %[[offset]], %[[4]], %[[offset_3]], %[[6]], %[[offset_8]], %[[c2_i64]])

// -----
#map = affine_map<(d0, d1, d2) -> (d1, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
module {
  func.func @gemm_2(%arg0: memref<64x32xf32>, %arg1: memref<32x64xf32>, %arg2: memref<64x64xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<64x32xf32>, vector<64x32xf32>
    %1 = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : memref<32x64xf32>, vector<32x64xf32>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<64x64xf32>, vector<64x64xf32>
    %3 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel"], kind = #vector.kind<add>} %0, %1, %2 : vector<64x32xf32>, vector<32x64xf32> into vector<64x64xf32>
    vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<64x64xf32>, memref<64x64xf32>
    return
  }
}
// CHECK-LABEL:  func.func @gemm_2(
// CHECK: %[[arg0:.*]]: memref<2x2x2x4xf32>, %[[arg1:.*]]: memref<2x4x8x2xf32>, %[[arg2:.*]]: memref<2x2x8x2xf32>) {
// CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[c2_i64:.*]] = arith.constant 2 : i64
// CHECK-DAG: %[[c4_i64:.*]] = arith.constant 4 : i64
// CHECK-DAG: %[[c8_i64:.*]] = arith.constant 8 : i64
// CHECK-DAG: %[[c16_i64:.*]] = arith.constant 16 : i64
// CHECK-DAG: %[[c64_i64:.*]] = arith.constant 64 : i64
// CHECK-DAG: %[[c0_i64:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[dispatch:.*]] = call @xsmm_brgemm_dispatch(%[[c1_i64]], %[[c2_i64]], %[[c2_i64]], %[[c4_i64]], %[[c8_i64]], %[[c16_i64]], %[[c16_i64]], %[[c4_i64]], %[[c64_i64]], %[[c0_i64]])
// CHECK: scf.forall (%[[arg3:.*]], %[[arg4:.*]]) in (2, 8) {
// CHECK-DAG:      %[[subview:.*]] = memref.subview %[[arg0]][%[[arg3]], 0, 0, 0] [1, 2, 2, 4] [1, 1, 1, 1]
// CHECK-DAG:      %[[subview_0:.*]] = memref.subview %[[arg1]][0, 0, %[[arg4]], 0] [2, 4, 1, 2] [1, 1, 1, 1]
// CHECK-DAG:      %[[subview_1:.*]] = memref.subview %[[arg2]][%[[arg3]], 0, %[[arg4]], 0] [1, 2, 1, 2] [1, 1, 1, 1]
// CHECK-DAG:      %[[base_buffer:.*]], %[[offset:.*]], %[[sizes:.*]]:3, %[[strides:.*]]:3 = memref.extract_strided_metadata %[[subview]]
// CHECK:          %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[subview]]
// CHECK-NEXT:     %[[cast:.*]] = arith.index_cast %intptr : index to i64
// CHECK-NEXT:     %[[inttoptr:.*]] = llvm.inttoptr %[[cast]]
// CHECK-DAG:      %[[base_buffer_2:.*]], %[[offset_3:.*]], %[[sizes_4:.*]]:3, %[[strides_5:.*]]:3 = memref.extract_strided_metadata %[[subview_0]]
// CHECK-DAG:      %[[intptr_6:.*]] = memref.extract_aligned_pointer_as_index %[[subview_0]]
// CHECK-NEXT:     %[[cast2:.*]] = arith.index_cast %[[intptr_6]]
// CHECK-NEXT:     %[[inttoptr2:.*]] = llvm.inttoptr %[[cast2]]
// CHECK-DAG:      %[[base_buffer_7:.*]], %[[offset_8:.*]], %[[sizes_9:.*]]:2, %[[strides_10:.*]]:2 = memref.extract_strided_metadata %[[subview_1]]
// CHECK-DAG:      %[[intptr_11:.*]] = memref.extract_aligned_pointer_as_index %[[subview_1]]
// CHECK-NEXT:     %[[cast3:.*]] = arith.index_cast %[[intptr_11]]
// CHECK-NEXT:     %[[inttoptr3:.*]] = llvm.inttoptr %[[cast3]]
// CHECK:          func.call @xsmm_brgemm_invoke(%[[c1_i64]], %[[0]], %[[2]], %[[offset]], %[[4]], %[[offset_3]], %[[6]], %[[offset_8]], %[[c2_i64]])

// -----
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d0)>
module {
  func.func @gemm_3(%arg0: memref<64x32xf32>, %arg1: memref<32x64xf32>, %arg2: memref<64x64xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<64x32xf32>, vector<64x32xf32>
    %1 = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : memref<32x64xf32>, vector<32x64xf32>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<64x64xf32>, vector<64x64xf32>
    %3 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %0, %1, %2 : vector<64x32xf32>, vector<32x64xf32> into vector<64x64xf32>
    vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<64x64xf32>, memref<64x64xf32>
    return
  }
}
// CHECK-LABEL:  func.func @gemm_3(
// CHECK: %[[arg0:.*]]: memref<2x2x2x4xf32>, %[[arg1:.*]]: memref<2x4x8x2xf32>, %[[arg2:.*]]: memref<2x2x8x2xf32>) {
// CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[c2_i64:.*]] = arith.constant 2 : i64
// CHECK-DAG: %[[c4_i64:.*]] = arith.constant 4 : i64
// CHECK-DAG: %[[c8_i64:.*]] = arith.constant 8 : i64
// CHECK-DAG: %[[c16_i64:.*]] = arith.constant 16 : i64
// CHECK-DAG: %[[c64_i64:.*]] = arith.constant 64 : i64
// CHECK-DAG: %[[c0_i64:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[dispatch:.*]] = call @xsmm_brgemm_dispatch(%[[c1_i64]], %[[c2_i64]], %[[c2_i64]], %[[c4_i64]], %[[c8_i64]], %[[c16_i64]], %[[c16_i64]], %[[c4_i64]], %[[c64_i64]], %[[c0_i64]])
// CHECK: scf.forall (%[[arg3:.*]], %[[arg4:.*]]) in (2, 8) {
// CHECK-DAG:      %[[subview:.*]] = memref.subview %[[arg0]][%[[arg3]], 0, 0, 0] [1, 2, 2, 4] [1, 1, 1, 1]
// CHECK-DAG:      %[[subview_0:.*]] = memref.subview %[[arg1]][0, 0, %[[arg4]], 0] [2, 4, 1, 2] [1, 1, 1, 1]
// CHECK-DAG:      %[[subview_1:.*]] = memref.subview %[[arg2]][%[[arg3]], 0, %[[arg4]], 0] [1, 2, 1, 2] [1, 1, 1, 1]
// CHECK-DAG:      %[[base_buffer:.*]], %[[offset:.*]], %[[sizes:.*]]:3, %[[strides:.*]]:3 = memref.extract_strided_metadata %[[subview]]
// CHECK:          %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[subview]]
// CHECK-NEXT:     %[[cast:.*]] = arith.index_cast %intptr : index to i64
// CHECK-NEXT:     %[[inttoptr:.*]] = llvm.inttoptr %[[cast]]
// CHECK-DAG:      %[[base_buffer_2:.*]], %[[offset_3:.*]], %[[sizes_4:.*]]:3, %[[strides_5:.*]]:3 = memref.extract_strided_metadata %[[subview_0]]
// CHECK-DAG:      %[[intptr_6:.*]] = memref.extract_aligned_pointer_as_index %[[subview_0]]
// CHECK-NEXT:     %[[cast2:.*]] = arith.index_cast %[[intptr_6]]
// CHECK-NEXT:     %[[inttoptr2:.*]] = llvm.inttoptr %[[cast2]]
// CHECK-DAG:      %[[base_buffer_7:.*]], %[[offset_8:.*]], %[[sizes_9:.*]]:2, %[[strides_10:.*]]:2 = memref.extract_strided_metadata %[[subview_1]]
// CHECK-DAG:      %[[intptr_11:.*]] = memref.extract_aligned_pointer_as_index %[[subview_1]]
// CHECK-NEXT:     %[[cast3:.*]] = arith.index_cast %[[intptr_11]]
// CHECK-NEXT:     %[[inttoptr3:.*]] = llvm.inttoptr %[[cast3]]
// CHECK:          func.call @xsmm_brgemm_invoke(%[[c1_i64]], %[[0]], %[[2]], %[[offset]], %[[4]], %[[offset_3]], %[[6]], %[[offset_8]], %[[c2_i64]])

// -----
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d0)>
module {
  func.func @gemm_4(%arg0: memref<64x32xf32>, %arg1: memref<64x32xf32>, %arg2: memref<64x64xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<64x32xf32>, vector<64x32xf32>
    %1 = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : memref<64x32xf32>, vector<64x32xf32>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<64x64xf32>, vector<64x64xf32>
    %3 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %0, %1, %2 : vector<64x32xf32>, vector<64x32xf32> into vector<64x64xf32>
    vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<64x64xf32>, memref<64x64xf32>
    return
  }
}
// CHECK-LABEL:  func.func @gemm_4(
// CHECK: %[[arg0:.*]]: memref<2x2x2x4xf32>, %[[arg1:.*]]: memref<2x4x8x2xf32>, %[[arg2:.*]]: memref<2x2x8x2xf32>) {
// CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[c2_i64:.*]] = arith.constant 2 : i64
// CHECK-DAG: %[[c4_i64:.*]] = arith.constant 4 : i64
// CHECK-DAG: %[[c8_i64:.*]] = arith.constant 8 : i64
// CHECK-DAG: %[[c16_i64:.*]] = arith.constant 16 : i64
// CHECK-DAG: %[[c64_i64:.*]] = arith.constant 64 : i64
// CHECK-DAG: %[[c0_i64:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[dispatch:.*]] = call @xsmm_brgemm_dispatch(%[[c1_i64]], %[[c2_i64]], %[[c2_i64]], %[[c4_i64]], %[[c8_i64]], %[[c16_i64]], %[[c16_i64]], %[[c4_i64]], %[[c64_i64]], %[[c0_i64]])
// CHECK: scf.forall (%[[arg3:.*]], %[[arg4:.*]]) in (2, 8) {
// CHECK-DAG:      %[[subview:.*]] = memref.subview %[[arg0]][%[[arg3]], 0, 0, 0] [1, 2, 2, 4] [1, 1, 1, 1]
// CHECK-DAG:      %[[subview_0:.*]] = memref.subview %[[arg1]][0, 0, %[[arg4]], 0] [2, 4, 1, 2] [1, 1, 1, 1]
// CHECK-DAG:      %[[subview_1:.*]] = memref.subview %[[arg2]][%[[arg3]], 0, %[[arg4]], 0] [1, 2, 1, 2] [1, 1, 1, 1]
// CHECK-DAG:      %[[base_buffer:.*]], %[[offset:.*]], %[[sizes:.*]]:3, %[[strides:.*]]:3 = memref.extract_strided_metadata %[[subview]]
// CHECK:          %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[subview]]
// CHECK-NEXT:     %[[cast:.*]] = arith.index_cast %intptr : index to i64
// CHECK-NEXT:     %[[inttoptr:.*]] = llvm.inttoptr %[[cast]]
// CHECK-DAG:      %[[base_buffer_2:.*]], %[[offset_3:.*]], %[[sizes_4:.*]]:3, %[[strides_5:.*]]:3 = memref.extract_strided_metadata %[[subview_0]]
// CHECK-DAG:      %[[intptr_6:.*]] = memref.extract_aligned_pointer_as_index %[[subview_0]]
// CHECK-NEXT:     %[[cast2:.*]] = arith.index_cast %[[intptr_6]]
// CHECK-NEXT:     %[[inttoptr2:.*]] = llvm.inttoptr %[[cast2]]
// CHECK-DAG:      %[[base_buffer_7:.*]], %[[offset_8:.*]], %[[sizes_9:.*]]:2, %[[strides_10:.*]]:2 = memref.extract_strided_metadata %[[subview_1]]
// CHECK-DAG:      %[[intptr_11:.*]] = memref.extract_aligned_pointer_as_index %[[subview_1]]
// CHECK-NEXT:     %[[cast3:.*]] = arith.index_cast %[[intptr_11]]
// CHECK-NEXT:     %[[inttoptr3:.*]] = llvm.inttoptr %[[cast3]]
// CHECK:          func.call @xsmm_brgemm_invoke(%[[c1_i64]], %[[0]], %[[2]], %[[offset]], %[[4]], %[[offset_3]], %[[6]], %[[offset_8]], %[[c2_i64]])

// -----
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
module {
  func.func @simple_brgemm(%arg0: memref<2x32x32xf32>, %arg1: memref<2x32x32xf32>, %arg2: memref<32x32xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<2x32x32xf32>, vector<2x32x32xf32>
    %1 = vector.transfer_read %arg1[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<2x32x32xf32>, vector<2x32x32xf32>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<32x32xf32>, vector<32x32xf32>
    %3 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %0, %1, %2 : vector<2x32x32xf32>, vector<2x32x32xf32> into vector<32x32xf32>
    vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xf32>, memref<32x32xf32>
    return
  }
}
// CHECK-LABEL:  func.func @simple_brgemm(
// CHECK: %[[arg0:.*]]: memref<2x2x2x4xf32>, %[[arg1:.*]]: memref<2x4x8x2xf32>, %[[arg2:.*]]: memref<2x2x8x2xf32>) {
// CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[c2_i64:.*]] = arith.constant 2 : i64
// CHECK-DAG: %[[c4_i64:.*]] = arith.constant 4 : i64
// CHECK-DAG: %[[c8_i64:.*]] = arith.constant 8 : i64
// CHECK-DAG: %[[c16_i64:.*]] = arith.constant 16 : i64
// CHECK-DAG: %[[c64_i64:.*]] = arith.constant 64 : i64
// CHECK-DAG: %[[c0_i64:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[dispatch:.*]] = call @xsmm_brgemm_dispatch(%[[c1_i64]], %[[c2_i64]], %[[c2_i64]], %[[c4_i64]], %[[c8_i64]], %[[c16_i64]], %[[c16_i64]], %[[c4_i64]], %[[c64_i64]], %[[c0_i64]])
// CHECK: scf.forall (%[[arg3:.*]], %[[arg4:.*]]) in (2, 8) {
// CHECK-DAG:      %[[subview:.*]] = memref.subview %[[arg0]][%[[arg3]], 0, 0, 0] [1, 2, 2, 4] [1, 1, 1, 1]
// CHECK-DAG:      %[[subview_0:.*]] = memref.subview %[[arg1]][0, 0, %[[arg4]], 0] [2, 4, 1, 2] [1, 1, 1, 1]
// CHECK-DAG:      %[[subview_1:.*]] = memref.subview %[[arg2]][%[[arg3]], 0, %[[arg4]], 0] [1, 2, 1, 2] [1, 1, 1, 1]
// CHECK-DAG:      %[[base_buffer:.*]], %[[offset:.*]], %[[sizes:.*]]:3, %[[strides:.*]]:3 = memref.extract_strided_metadata %[[subview]]
// CHECK:          %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[subview]]
// CHECK-NEXT:     %[[cast:.*]] = arith.index_cast %intptr : index to i64
// CHECK-NEXT:     %[[inttoptr:.*]] = llvm.inttoptr %[[cast]]
// CHECK-DAG:      %[[base_buffer_2:.*]], %[[offset_3:.*]], %[[sizes_4:.*]]:3, %[[strides_5:.*]]:3 = memref.extract_strided_metadata %[[subview_0]]
// CHECK-DAG:      %[[intptr_6:.*]] = memref.extract_aligned_pointer_as_index %[[subview_0]]
// CHECK-NEXT:     %[[cast2:.*]] = arith.index_cast %[[intptr_6]]
// CHECK-NEXT:     %[[inttoptr2:.*]] = llvm.inttoptr %[[cast2]]
// CHECK-DAG:      %[[base_buffer_7:.*]], %[[offset_8:.*]], %[[sizes_9:.*]]:2, %[[strides_10:.*]]:2 = memref.extract_strided_metadata %[[subview_1]]
// CHECK-DAG:      %[[intptr_11:.*]] = memref.extract_aligned_pointer_as_index %[[subview_1]]
// CHECK-NEXT:     %[[cast3:.*]] = arith.index_cast %[[intptr_11]]
// CHECK-NEXT:     %[[inttoptr3:.*]] = llvm.inttoptr %[[cast3]]
// CHECK:          func.call @xsmm_brgemm_invoke(%[[c1_i64]], %[[0]], %[[2]], %[[offset]], %[[4]], %[[offset_3]], %[[6]], %[[offset_8]], %[[c2_i64]])

// -----
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
module {
  func.func @vnni_brgemm_interchanged(%arg0: memref<16x32x32xbf16>, %arg1: memref<16x16x32x2xbf16>, %arg2: memref<32x32xbf16>) {
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
// CHECK-LABEL:  func.func @vnni_brgemm_interchanged(
// CHECK: %[[arg0:.*]]: memref<2x2x2x4xf32>, %[[arg1:.*]]: memref<2x4x8x2xf32>, %[[arg2:.*]]: memref<2x2x8x2xf32>) {
// CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[c2_i64:.*]] = arith.constant 2 : i64
// CHECK-DAG: %[[c4_i64:.*]] = arith.constant 4 : i64
// CHECK-DAG: %[[c8_i64:.*]] = arith.constant 8 : i64
// CHECK-DAG: %[[c16_i64:.*]] = arith.constant 16 : i64
// CHECK-DAG: %[[c64_i64:.*]] = arith.constant 64 : i64
// CHECK-DAG: %[[c0_i64:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[dispatch:.*]] = call @xsmm_brgemm_dispatch(%[[c1_i64]], %[[c2_i64]], %[[c2_i64]], %[[c4_i64]], %[[c8_i64]], %[[c16_i64]], %[[c16_i64]], %[[c4_i64]], %[[c64_i64]], %[[c0_i64]])
// CHECK: scf.forall (%[[arg3:.*]], %[[arg4:.*]]) in (2, 8) {
// CHECK-DAG:      %[[subview:.*]] = memref.subview %[[arg0]][%[[arg3]], 0, 0, 0] [1, 2, 2, 4] [1, 1, 1, 1]
// CHECK-DAG:      %[[subview_0:.*]] = memref.subview %[[arg1]][0, 0, %[[arg4]], 0] [2, 4, 1, 2] [1, 1, 1, 1]
// CHECK-DAG:      %[[subview_1:.*]] = memref.subview %[[arg2]][%[[arg3]], 0, %[[arg4]], 0] [1, 2, 1, 2] [1, 1, 1, 1]
// CHECK-DAG:      %[[base_buffer:.*]], %[[offset:.*]], %[[sizes:.*]]:3, %[[strides:.*]]:3 = memref.extract_strided_metadata %[[subview]]
// CHECK:          %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[subview]]
// CHECK-NEXT:     %[[cast:.*]] = arith.index_cast %intptr : index to i64
// CHECK-NEXT:     %[[inttoptr:.*]] = llvm.inttoptr %[[cast]]
// CHECK-DAG:      %[[base_buffer_2:.*]], %[[offset_3:.*]], %[[sizes_4:.*]]:3, %[[strides_5:.*]]:3 = memref.extract_strided_metadata %[[subview_0]]
// CHECK-DAG:      %[[intptr_6:.*]] = memref.extract_aligned_pointer_as_index %[[subview_0]]
// CHECK-NEXT:     %[[cast2:.*]] = arith.index_cast %[[intptr_6]]
// CHECK-NEXT:     %[[inttoptr2:.*]] = llvm.inttoptr %[[cast2]]
// CHECK-DAG:      %[[base_buffer_7:.*]], %[[offset_8:.*]], %[[sizes_9:.*]]:2, %[[strides_10:.*]]:2 = memref.extract_strided_metadata %[[subview_1]]
// CHECK-DAG:      %[[intptr_11:.*]] = memref.extract_aligned_pointer_as_index %[[subview_1]]
// CHECK-NEXT:     %[[cast3:.*]] = arith.index_cast %[[intptr_11]]
// CHECK-NEXT:     %[[inttoptr3:.*]] = llvm.inttoptr %[[cast3]]
// CHECK:          func.call @xsmm_brgemm_invoke(%[[c1_i64]], %[[0]], %[[2]], %[[offset]], %[[4]], %[[offset_3]], %[[6]], %[[offset_8]], %[[c2_i64]])

// -----
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2)>
module {
  func.func @vnni_brgemm(%arg0: memref<16x32x32xbf16>, %arg1: memref<16x16x32x2xbf16>, %arg2: memref<32x32xbf16>) {
    %cst = arith.constant 0.000000e+00 : bf16
    %c0 = arith.constant 0 : index
    %expand_shape = memref.expand_shape %arg0 [[0], [1], [2, 3]] output_shape [16, 32, 16, 2] : memref<16x32x32xbf16> into memref<16x32x16x2xbf16>
    %0 = vector.transfer_read %expand_shape[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<16x32x16x2xbf16>, vector<16x32x16x2xbf16>
    %1 = vector.transfer_read %arg1[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<16x16x32x2xbf16>, vector<16x16x32x2xbf16>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<32x32xbf16>, vector<32x32xbf16>
    %3 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction", "reduction"], kind = #vector.kind<add>} %0, %1, %2 : vector<16x32x16x2xbf16>, vector<16x16x32x2xbf16> into vector<32x32xbf16>
    vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xbf16>, memref<32x32xbf16>
    return
  }
}
// CHECK-LABEL:  func.func @vnni_brgemm(
// CHECK: %[[arg0:.*]]: memref<2x2x2x4xf32>, %[[arg1:.*]]: memref<2x4x8x2xf32>, %[[arg2:.*]]: memref<2x2x8x2xf32>) {
// CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[c2_i64:.*]] = arith.constant 2 : i64
// CHECK-DAG: %[[c4_i64:.*]] = arith.constant 4 : i64
// CHECK-DAG: %[[c8_i64:.*]] = arith.constant 8 : i64
// CHECK-DAG: %[[c16_i64:.*]] = arith.constant 16 : i64
// CHECK-DAG: %[[c64_i64:.*]] = arith.constant 64 : i64
// CHECK-DAG: %[[c0_i64:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[dispatch:.*]] = call @xsmm_brgemm_dispatch(%[[c1_i64]], %[[c2_i64]], %[[c2_i64]], %[[c4_i64]], %[[c8_i64]], %[[c16_i64]], %[[c16_i64]], %[[c4_i64]], %[[c64_i64]], %[[c0_i64]])
// CHECK: scf.forall (%[[arg3:.*]], %[[arg4:.*]]) in (2, 8) {
// CHECK-DAG:      %[[subview:.*]] = memref.subview %[[arg0]][%[[arg3]], 0, 0, 0] [1, 2, 2, 4] [1, 1, 1, 1]
// CHECK-DAG:      %[[subview_0:.*]] = memref.subview %[[arg1]][0, 0, %[[arg4]], 0] [2, 4, 1, 2] [1, 1, 1, 1]
// CHECK-DAG:      %[[subview_1:.*]] = memref.subview %[[arg2]][%[[arg3]], 0, %[[arg4]], 0] [1, 2, 1, 2] [1, 1, 1, 1]
// CHECK-DAG:      %[[base_buffer:.*]], %[[offset:.*]], %[[sizes:.*]]:3, %[[strides:.*]]:3 = memref.extract_strided_metadata %[[subview]]
// CHECK:          %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[subview]]
// CHECK-NEXT:     %[[cast:.*]] = arith.index_cast %intptr : index to i64
// CHECK-NEXT:     %[[inttoptr:.*]] = llvm.inttoptr %[[cast]]
// CHECK-DAG:      %[[base_buffer_2:.*]], %[[offset_3:.*]], %[[sizes_4:.*]]:3, %[[strides_5:.*]]:3 = memref.extract_strided_metadata %[[subview_0]]
// CHECK-DAG:      %[[intptr_6:.*]] = memref.extract_aligned_pointer_as_index %[[subview_0]]
// CHECK-NEXT:     %[[cast2:.*]] = arith.index_cast %[[intptr_6]]
// CHECK-NEXT:     %[[inttoptr2:.*]] = llvm.inttoptr %[[cast2]]
// CHECK-DAG:      %[[base_buffer_7:.*]], %[[offset_8:.*]], %[[sizes_9:.*]]:2, %[[strides_10:.*]]:2 = memref.extract_strided_metadata %[[subview_1]]
// CHECK-DAG:      %[[intptr_11:.*]] = memref.extract_aligned_pointer_as_index %[[subview_1]]
// CHECK-NEXT:     %[[cast3:.*]] = arith.index_cast %[[intptr_11]]
// CHECK-NEXT:     %[[inttoptr3:.*]] = llvm.inttoptr %[[cast3]]
// CHECK:          func.call @xsmm_brgemm_invoke(%[[c1_i64]], %[[0]], %[[2]], %[[offset]], %[[4]], %[[offset_3]], %[[6]], %[[offset_8]], %[[c2_i64]])

// -----
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2)>
module {
  func.func @vnni_brgemm_strided(%arg0: memref<8x8x8xbf16, strided<[64, 8, 1], offset: ?>>, %arg1: memref<8x4x8x2xbf16, strided<[64, 16, 2, 1], offset: ?>>, %arg2: memref<8x8xbf16>) {
    %cst = arith.constant 0.000000e+00 : bf16
    %c0 = arith.constant 0 : index
    %expand_shape = memref.expand_shape %arg0 [[0], [1], [2, 3]] output_shape [8, 8, 4, 2] : memref<8x8x8xbf16, strided<[64, 8, 1], offset: ?>> into memref<8x8x4x2xbf16, strided<[64, 8, 2, 1], offset: ?>>
    %0 = vector.transfer_read %expand_shape[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<8x8x4x2xbf16, strided<[64, 8, 2, 1], offset: ?>>, vector<8x8x4x2xbf16>
    %1 = vector.transfer_read %arg1[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<8x4x8x2xbf16, strided<[64, 16, 2, 1], offset: ?>>, vector<8x4x8x2xbf16>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<8x8xbf16>, vector<8x8xbf16>
    %3 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction", "reduction"], kind = #vector.kind<add>} %0, %1, %2 : vector<8x8x4x2xbf16>, vector<8x4x8x2xbf16> into vector<8x8xbf16>
    vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<8x8xbf16>, memref<8x8xbf16>
    return
  }
}
// CHECK-LABEL:  func.func @vnni_brgemm_strided(
// CHECK: %[[arg0:.*]]: memref<2x2x2x4xf32>, %[[arg1:.*]]: memref<2x4x8x2xf32>, %[[arg2:.*]]: memref<2x2x8x2xf32>) {
// CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[c2_i64:.*]] = arith.constant 2 : i64
// CHECK-DAG: %[[c4_i64:.*]] = arith.constant 4 : i64
// CHECK-DAG: %[[c8_i64:.*]] = arith.constant 8 : i64
// CHECK-DAG: %[[c16_i64:.*]] = arith.constant 16 : i64
// CHECK-DAG: %[[c64_i64:.*]] = arith.constant 64 : i64
// CHECK-DAG: %[[c0_i64:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[dispatch:.*]] = call @xsmm_brgemm_dispatch(%[[c1_i64]], %[[c2_i64]], %[[c2_i64]], %[[c4_i64]], %[[c8_i64]], %[[c16_i64]], %[[c16_i64]], %[[c4_i64]], %[[c64_i64]], %[[c0_i64]])
// CHECK: scf.forall (%[[arg3:.*]], %[[arg4:.*]]) in (2, 8) {
// CHECK-DAG:      %[[subview:.*]] = memref.subview %[[arg0]][%[[arg3]], 0, 0, 0] [1, 2, 2, 4] [1, 1, 1, 1]
// CHECK-DAG:      %[[subview_0:.*]] = memref.subview %[[arg1]][0, 0, %[[arg4]], 0] [2, 4, 1, 2] [1, 1, 1, 1]
// CHECK-DAG:      %[[subview_1:.*]] = memref.subview %[[arg2]][%[[arg3]], 0, %[[arg4]], 0] [1, 2, 1, 2] [1, 1, 1, 1]
// CHECK-DAG:      %[[base_buffer:.*]], %[[offset:.*]], %[[sizes:.*]]:3, %[[strides:.*]]:3 = memref.extract_strided_metadata %[[subview]]
// CHECK:          %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[subview]]
// CHECK-NEXT:     %[[cast:.*]] = arith.index_cast %intptr : index to i64
// CHECK-NEXT:     %[[inttoptr:.*]] = llvm.inttoptr %[[cast]]
// CHECK-DAG:      %[[base_buffer_2:.*]], %[[offset_3:.*]], %[[sizes_4:.*]]:3, %[[strides_5:.*]]:3 = memref.extract_strided_metadata %[[subview_0]]
// CHECK-DAG:      %[[intptr_6:.*]] = memref.extract_aligned_pointer_as_index %[[subview_0]]
// CHECK-NEXT:     %[[cast2:.*]] = arith.index_cast %[[intptr_6]]
// CHECK-NEXT:     %[[inttoptr2:.*]] = llvm.inttoptr %[[cast2]]
// CHECK-DAG:      %[[base_buffer_7:.*]], %[[offset_8:.*]], %[[sizes_9:.*]]:2, %[[strides_10:.*]]:2 = memref.extract_strided_metadata %[[subview_1]]
// CHECK-DAG:      %[[intptr_11:.*]] = memref.extract_aligned_pointer_as_index %[[subview_1]]
// CHECK-NEXT:     %[[cast3:.*]] = arith.index_cast %[[intptr_11]]
// CHECK-NEXT:     %[[inttoptr3:.*]] = llvm.inttoptr %[[cast3]]
// CHECK:          func.call @xsmm_brgemm_invoke(%[[c1_i64]], %[[0]], %[[2]], %[[offset]], %[[4]], %[[offset_3]], %[[6]], %[[offset_8]], %[[c2_i64]])

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
// CHECK-LABEL:  func.func @vnni_brgemm_require_transpose_on_C(
// CHECK: %[[arg0:.*]]: memref<2x2x2x4xf32>, %[[arg1:.*]]: memref<2x4x8x2xf32>, %[[arg2:.*]]: memref<2x2x8x2xf32>) {
// CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[c2_i64:.*]] = arith.constant 2 : i64
// CHECK-DAG: %[[c4_i64:.*]] = arith.constant 4 : i64
// CHECK-DAG: %[[c8_i64:.*]] = arith.constant 8 : i64
// CHECK-DAG: %[[c16_i64:.*]] = arith.constant 16 : i64
// CHECK-DAG: %[[c64_i64:.*]] = arith.constant 64 : i64
// CHECK-DAG: %[[c0_i64:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[dispatch:.*]] = call @xsmm_brgemm_dispatch(%[[c1_i64]], %[[c2_i64]], %[[c2_i64]], %[[c4_i64]], %[[c8_i64]], %[[c16_i64]], %[[c16_i64]], %[[c4_i64]], %[[c64_i64]], %[[c0_i64]])
// CHECK: scf.forall (%[[arg3:.*]], %[[arg4:.*]]) in (2, 8) {
// CHECK-DAG:      %[[subview:.*]] = memref.subview %[[arg0]][%[[arg3]], 0, 0, 0] [1, 2, 2, 4] [1, 1, 1, 1]
// CHECK-DAG:      %[[subview_0:.*]] = memref.subview %[[arg1]][0, 0, %[[arg4]], 0] [2, 4, 1, 2] [1, 1, 1, 1]
// CHECK-DAG:      %[[subview_1:.*]] = memref.subview %[[arg2]][%[[arg3]], 0, %[[arg4]], 0] [1, 2, 1, 2] [1, 1, 1, 1]
// CHECK-DAG:      %[[base_buffer:.*]], %[[offset:.*]], %[[sizes:.*]]:3, %[[strides:.*]]:3 = memref.extract_strided_metadata %[[subview]]
// CHECK:          %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[subview]]
// CHECK-NEXT:     %[[cast:.*]] = arith.index_cast %intptr : index to i64
// CHECK-NEXT:     %[[inttoptr:.*]] = llvm.inttoptr %[[cast]]
// CHECK-DAG:      %[[base_buffer_2:.*]], %[[offset_3:.*]], %[[sizes_4:.*]]:3, %[[strides_5:.*]]:3 = memref.extract_strided_metadata %[[subview_0]]
// CHECK-DAG:      %[[intptr_6:.*]] = memref.extract_aligned_pointer_as_index %[[subview_0]]
// CHECK-NEXT:     %[[cast2:.*]] = arith.index_cast %[[intptr_6]]
// CHECK-NEXT:     %[[inttoptr2:.*]] = llvm.inttoptr %[[cast2]]
// CHECK-DAG:      %[[base_buffer_7:.*]], %[[offset_8:.*]], %[[sizes_9:.*]]:2, %[[strides_10:.*]]:2 = memref.extract_strided_metadata %[[subview_1]]
// CHECK-DAG:      %[[intptr_11:.*]] = memref.extract_aligned_pointer_as_index %[[subview_1]]
// CHECK-NEXT:     %[[cast3:.*]] = arith.index_cast %[[intptr_11]]
// CHECK-NEXT:     %[[inttoptr3:.*]] = llvm.inttoptr %[[cast3]]
// CHECK:          func.call @xsmm_brgemm_invoke(%[[c1_i64]], %[[0]], %[[2]], %[[offset]], %[[4]], %[[offset_3]], %[[6]], %[[offset_8]], %[[c2_i64]])

// -----
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d2)>
module {
  func.func @brgemm_not_vnni(%arg0: memref<16x32x32xbf16>, %arg1: memref<16x16x32x2xbf16>, %arg2: memref<32x32xbf16>) {
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
// CHECK-LABEL:  func.func @brgemm_not_vnni(
// CHECK: %[[arg0:.*]]: memref<2x2x2x4xf32>, %[[arg1:.*]]: memref<2x4x8x2xf32>, %[[arg2:.*]]: memref<2x2x8x2xf32>) {
// CHECK-NOT: %[[dispatch:.*]] = call @xsmm_brgemm_dispatch(%[[c1_i64]], %[[c2_i64]], %[[c2_i64]], %[[c4_i64]], %[[c8_i64]], %[[c16_i64]], %[[c16_i64]], %[[c4_i64]], %[[c64_i64]], %[[c0_i64]])
// CHECK-NOT: func.call @xsmm_brgemm_invoke(%[[c1_i64]], %[[0]], %[[2]], %[[offset]], %[[4]], %[[offset_3]], %[[6]], %[[offset_8]], %[[c2_i64]])

