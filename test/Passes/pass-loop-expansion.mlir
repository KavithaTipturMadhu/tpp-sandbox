// RUN: tpp-opt --loop-expansion-pass="num-outer-parallel=2" --canonicalize %s | FileCheck %s

module {
  func.func @entry(%arg0: memref<8x4x4x4xbf16>, %arg1: memref<4x4x2x4x2xbf16>, %arg2: memref<8x4x4x4xbf16>) {
    %c4_i64 = arith.constant 4 : i64
    %expand_shape = memref.expand_shape %arg0 [[0, 1], [2], [3], [4]] output_shape [2, 4, 4, 4, 4] : memref<8x4x4x4xbf16> into memref<2x4x4x4x4xbf16>
    %expand_shape_0 = memref.expand_shape %arg1 [[0, 1], [2], [3], [4], [5]] output_shape [2, 2, 4, 2, 4, 2] : memref<4x4x2x4x2xbf16> into memref<2x2x4x2x4x2xbf16>
    %expand_shape_1 = memref.expand_shape %arg2 [[0, 1], [2, 3], [4], [5]] output_shape [2, 4, 2, 2, 4, 4] : memref<8x4x4x4xbf16> into memref<2x4x2x2x4x4xbf16>
    scf.forall (%arg3, %arg4, %arg5, %arg6) in (2, 4, 2, 2) {
      %subview = memref.subview %expand_shape_1[%arg3, %arg4, %arg5, %arg6, 0, 0] [1, 1, 1, 1, 4, 4] [1, 1, 1, 1, 1, 1] : memref<2x4x2x2x4x4xbf16> to memref<4x4xbf16, strided<[4, 1], offset: ?>>
      %subview_2 = memref.subview %expand_shape_0[%arg5, %arg6, 0, 0, 0, 0] [1, 1, 4, 2, 4, 2] [1, 1, 1, 1, 1, 1] : memref<2x2x4x2x4x2xbf16> to memref<4x2x4x2xbf16, strided<[16, 8, 2, 1], offset: ?>>
      %subview_3 = memref.subview %expand_shape[%arg3, %arg4, 0, 0, 0] [1, 1, 4, 4, 4] [1, 1, 1, 1, 1] : memref<2x4x4x4x4xbf16> to memref<4x4x4xbf16, strided<[16, 4, 1], offset: ?>>
      %0 = xsmm.brgemm.dispatch [4, 4, 4, 4, 4, 4, 16, 16] flags = (vnni_b) data_type = bf16
      xsmm.brgemm(data_type = bf16, %0, %subview_3, %subview_2, %subview, %c4_i64) : (i64, memref<4x4x4xbf16, strided<[16, 4, 1], offset: ?>>, memref<4x2x4x2xbf16, strided<[16, 8, 2, 1], offset: ?>>, memref<4x4xbf16, strided<[4, 1], offset: ?>>, i64) -> ()
    }
    return
  }
}

// CHECK-LABEL: func.func @entry
// CHECK: (%[[ARG0:.*]]: memref<8x4x4x4xbf16>, %[[ARG1:.*]]: memref<4x4x2x4x2xbf16>, %[[ARG2:.*]]: memref<8x4x4x4xbf16>) {
// CHECK-DAG: %[[c4_i64:.*]] = arith.constant 4 : i64
// CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[c2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[expand_shape:.*]] = memref.expand_shape %[[ARG0]] {{\[}}[0, 1], [2], [3], [4]] output_shape [2, 4, 4, 4, 4] : memref<8x4x4x4xbf16> into memref<2x4x4x4x4xbf16>
// CHECK-DAG: %[[expand_shape_0:.*]] = memref.expand_shape %[[ARG1]] {{\[}}[0, 1], [2], [3], [4], [5]] output_shape [2, 2, 4, 2, 4, 2] : memref<4x4x2x4x2xbf16> into memref<2x2x4x2x4x2xbf16>
// CHECK-DAG: %[[expand_shape_1:.*]] = memref.expand_shape %[[ARG2]] {{\[}}[0, 1], [2, 3], [4], [5]] output_shape [2, 4, 2, 2, 4, 4] : memref<8x4x4x4xbf16> into memref<2x4x2x2x4x4xbf16>
// CHECK:     scf.parallel (%[[ARG3:.*]], %[[ARG4:.*]]) = (%[[c0]], %[[c0]]) to (%[[c2]], %[[c4]]) step (%[[c1]], %[[c1]]) {
// CHECK:      scf.for %[[ARG5:.*]] = %[[c0]] to %[[c2]] step %[[c1]] {
// CHECK:        scf.for %[[ARG6:.*]] = %[[c0]] to %[[c2]] step %[[c1]] {
// CHECK-DAG:        %[[subview:.*]] = memref.subview %[[expand_shape_1]][%[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]], 0, 0] [1, 1, 1, 1, 4, 4] [1, 1, 1, 1, 1, 1] : memref<2x4x2x2x4x4xbf16> to memref<4x4xbf16, strided<[4, 1], offset: ?>>
// CHECK-DAG:        %[[subview_2:.*]] = memref.subview %[[expand_shape_0]][%[[ARG5]], %[[ARG6]], 0, 0, 0, 0] [1, 1, 4, 2, 4, 2] [1, 1, 1, 1, 1, 1] : memref<2x2x4x2x4x2xbf16> to memref<4x2x4x2xbf16, strided<[16, 8, 2, 1], offset: ?>>
// CHECK-DAG:        %[[subview_3:.*]] = memref.subview %[[expand_shape]][%[[ARG3]], %[[ARG4]], 0, 0, 0] [1, 1, 4, 4, 4] [1, 1, 1, 1, 1] : memref<2x4x4x4x4xbf16> to memref<4x4x4xbf16, strided<[16, 4, 1], offset: ?>>
// CHECK:          %[[DISPATCH:.*]] = xsmm.brgemm.dispatch [4, 4, 4, 4, 4, 4, 16, 16] flags = (vnni_b) data_type = bf16
// CHECK:          xsmm.brgemm(data_type = bf16, %[[DISPATCH]], %[[subview_3]], %[[subview_2]], %[[subview]], %[[c4_i64]]) : (i64, memref<4x4x4xbf16, strided<[16, 4, 1], offset: ?>>, memref<4x2x4x2xbf16, strided<[16, 8, 2, 1], offset: ?>>, memref<4x4xbf16, strided<[4, 1], offset: ?>>, i64) -> ()
// CHECK:        }
// CHECK:      }
// CHECK:      scf.reduce
