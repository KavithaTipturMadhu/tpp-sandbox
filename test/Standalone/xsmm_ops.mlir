// RUN: standalone-opt %s | standalone-opt | FileCheck %s

// CHECK-LABEL: @myfunc
func.func @myfunc(%arg0: memref<2x2xf32>, 
                  %arg1: memref<2x2xf32>, %arg2: memref<2x2xf32>) -> memref<2x2xf32> {

  %c3_i64 = arith.constant 3 : i64
  // CHECK: xsmm.ternary
  xsmm.ternary @libxsmm_matmul(%c3_i64, %arg0, %arg1, %arg2) 
    : (i64, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()

  // CHECK: xsmm.binary
  xsmm.binary @libxsmm_add(%arg0, %arg1)
    : (memref<2x2xf32>, memref<2x2xf32>) -> ()

  // CHECK: xsmm.unary
  xsmm.unary @libxsmm_relu(%arg0)
    : (memref<2x2xf32>) -> ()

  // CHECK: xsmm.ternary.dispatch
  xsmm.ternary.dispatch matmul [3 : i64, 2 : i64, 1 : i64]

  // CHECK: xsmm.binary.dispatch
  xsmm.binary.dispatch add [3 : i64, 2 : i64, 1 : i64]

  // CHECK: xsmm.unary.dispatch
  xsmm.unary.dispatch identity [3 : i64, 2 : i64, 1 : i64] (bcast_row)

  return %arg2: memref<2x2xf32>
}
