// RUN: tpp-opt %s -default-tpp-passes -split-input-file | FileCheck %s

// CHECK: func.func @matmul(
// CHECK-SAME:  %[[ARG0:.+]]: memref<4x8xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<8x4xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<4x4xf32>)
func.func @matmul(%A: tensor<4x8xf32>,
          %B: tensor<8x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: call @xsmm_matmul_dispatch
  // CHECK: %[[cast0:.*]] = memref.cast %[[ARG0]]
  // CHECK: %[[cast1:.*]] = memref.cast %[[ARG1]]
  // CHECK: %[[cast2:.*]] = memref.cast %[[ARG2]]
  // CHECK: call @xsmm_matmul_invoke({{.*}}%[[cast0]], %[[cast1]], %[[cast2]]
  %D = linalg.matmul ins(%A, %B: tensor<4x8xf32>, tensor<8x4xf32>) outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>

  return %D : tensor<4x4xf32>
}

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-LABEL: func.func @blocked_matmul(
// CHECK-SAME: %[[ARG0:.+]]: memref<4x16x32x32xf32>,
// CHECK-SAME: %[[ARG1:.+]]: memref<8x16x32x32xf32>,
// CHECK-SAME: %[[ARG2:.+]]: memref<4x8x32x32xf32>)
func.func @blocked_matmul(%arg0: tensor<4x16x32x32xf32>, %arg1: tensor<8x16x32x32xf32>, %arg2: tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> {
  // CHECK: call @xsmm_brgemm_dispatch
  // CHECK: scf.parallel
  // CHECK:   %[[cast:.*]] = memref.cast
  // CHECK:   %[[cast1:.*]] = memref.cast
  // CHECK:   %[[cast2:.*]] = memref.cast
  // CHECK:   call @xsmm_brgemm_invoke({{.*}}%[[cast]], %[[cast1]], %[[cast2]]
  %1 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%arg2 : tensor<4x8x32x32xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %8 = arith.mulf %arg3, %arg4 : f32
      %9 = arith.addf %arg5, %8 : f32
      linalg.yield %9 : f32
    } -> tensor<4x8x32x32xf32>

  return %1 :  tensor<4x8x32x32xf32>
}

// -----

// 1x1 Conv2D shapes
!conv1x1_input_tensor_t  = tensor<1x7x7x2048xf32> // N,H,W,Ic
!conv1x1_filter_tensor_t = tensor<1x1x2048x512xf32> // H,W,Ic,Oc
!conv1x1_output_tensor_t = tensor<1x7x7x512xf32> // N,H,W,Oc

// CHECK-LABEL: @conv2d_1x1(
// CHECK-SAME: %[[arg:.*]]: memref<1x7x7x2048xf32>) -> memref<1x7x7x512xf32> {
func.func @conv2d_1x1(
      %arg0 : !conv1x1_input_tensor_t) -> !conv1x1_output_tensor_t {
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_9 = arith.constant dense<0.000000e+00> : !conv1x1_output_tensor_t

  // Conv2D weights
  %cst = arith.constant dense<0.00332225906> : !conv1x1_filter_tensor_t

  // 1x1 Conv2D
  // CHECK: call @xsmm_matmul_dispatch
  // CHECK: %[[cast:.*]] = memref.cast
  // CHECK: %[[cast1:.*]] = memref.cast
  // CHECK: %[[cast2:.*]] = memref.cast
  // CHECK: call @xsmm_matmul_invoke({{.*}}%[[cast]], %[[cast1]], %[[cast2]]
  %0 = tensor.empty() : !conv1x1_output_tensor_t
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : !conv1x1_output_tensor_t) -> !conv1x1_output_tensor_t
  %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
              ins(%arg0, %cst : !conv1x1_input_tensor_t, !conv1x1_filter_tensor_t)
              outs(%1 : !conv1x1_output_tensor_t) -> !conv1x1_output_tensor_t

  // CHECK: return {{.*}} : memref<1x7x7x512xf32>
  return %2 : !conv1x1_output_tensor_t
}

// -----

#map = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-LABEL: @conv2d_1x1_decomposed(
// CHECK-SAME: %[[arg:.*]]: memref<1x7x7x2048xf32>) -> memref<1x7x7x512xf32> {
func.func @conv2d_1x1_decomposed(
      %arg0 : tensor<1x7x7x2048xf32>) -> tensor<1x7x7x512xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c7 = arith.constant 7 : index

  // Conv2D weights
  %cst = arith.constant dense<0.00332225906> : tensor<2048x512xf32>

  // 1x1 Conv2D
  // CHECK: call @xsmm_matmul_dispatch
  // CHECK: scf.for
  // CHECK:   %[[cast:.*]] = memref.cast
  // CHECK:   %[[cast1:.*]] = memref.cast
  // CHECK:   %[[cast2:.*]] = memref.cast
  // CHECK:   call @xsmm_matmul_invoke({{.*}}%[[cast]], %[[cast1]], %[[cast2]]
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x7x7x512xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
  %2 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %1) -> (tensor<1x7x7x512xf32>) {
    %3 = scf.for %arg3 = %c0 to %c7 step %c1 iter_args(%arg4 = %arg2) -> (tensor<1x7x7x512xf32>) {
      %4 = scf.for %arg5 = %c0 to %c1 step %c1 iter_args(%arg6 = %arg4) -> (tensor<1x7x7x512xf32>) {
        %5 = scf.for %arg7 = %c0 to %c1 step %c1 iter_args(%arg8 = %arg6) -> (tensor<1x7x7x512xf32>) {
          %6 = affine.apply #map(%arg3, %arg5)
          %extracted_slice = tensor.extract_slice %arg0[%arg1, %6, %arg7, 0] [1, 1, 7, 2048] [1, 1, 1, 1] : tensor<1x7x7x2048xf32> to tensor<7x2048xf32>
          %extracted_slice_1 = tensor.extract_slice %arg8[%arg1, %arg3, 0, 0] [1, 1, 7, 512] [1, 1, 1, 1] : tensor<1x7x7x512xf32> to tensor<7x512xf32>
          %7 = linalg.matmul ins(%extracted_slice, %cst : tensor<7x2048xf32>, tensor<2048x512xf32>) outs(%extracted_slice_1 : tensor<7x512xf32>) -> tensor<7x512xf32>
          %inserted_slice = tensor.insert_slice %7 into %arg8[%arg1, %arg3, 0, 0] [1, 1, 7, 512] [1, 1, 1, 1] : tensor<7x512xf32> into tensor<1x7x7x512xf32>
          scf.yield %inserted_slice : tensor<1x7x7x512xf32>
        }
        scf.yield %5 : tensor<1x7x7x512xf32>
      }
      scf.yield %4 : tensor<1x7x7x512xf32>
    }
    scf.yield %3 : tensor<1x7x7x512xf32>
  }

  // CHECK: return {{.*}} : memref<1x7x7x512xf32>
  return %2 : tensor<1x7x7x512xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func.func @mlp(
// CHECK-SAME:  %[[ARG0:.+]]: memref<128x256xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<256x512xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<512xf32>,
// CHECK-SAME:  %[[ARG3:.+]]: memref<128x512xf32>)
func.func @mlp(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>,
  %arg2: tensor<512xf32>,  %output: tensor<128x512xf32>) -> tensor<128x512xf32> {

  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C128:.+]] = arith.constant 128 : index
  // CHECK-DAG: %[[C512:.+]] = arith.constant 512 : index
  // CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index

  // CHECK: call @xsmm_unary_dispatch
  // CHECK-DAG: call @xsmm_matmul_dispatch
  // CHECK-DAG: call @xsmm_unary_dispatch
  
  // CHECK: scf.parallel (%[[I:.+]], %[[J:.+]]) = 
  // CHECK-SAME: (%[[C0]], %[[C0]]) to (%[[C128]], %[[C512]]) step (%[[C32]], %[[C32]])
  // CHECK: %[[SUB:.+]] = memref.subview %[[ARG0]]
  // Identity
  // CHECK: %[[SUB0:.+]] = memref.subview %[[ARG1]]
  // CHECK: %[[SUB1:.+]] = memref.subview %[[ARG2]]
  // CHECK: %[[SUB2:.+]] = memref.subview %[[ARG3]]
  // CHECK: %[[cast:.*]] = memref.cast %[[SUB1]]
  // CHECK: %[[cast0:.*]] = memref.cast %[[SUB2]]
  // CHECK: call @xsmm_unary_invoke({{.*}}%[[cast]], %[[cast0]]
  %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<512xf32>) outs(%output : tensor<128x512xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
  } -> tensor<128x512xf32>

  // Matmul
  // CHECK: %[[cast1:.*]] = memref.cast %[[SUB]]
  // CHECK: %[[cast2:.*]] = memref.cast %[[SUB0]]
  // CHECK: call @xsmm_matmul_invoke({{.*}}%[[cast1]], %[[cast2]], %[[cast0]]
  %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>) outs(%1 : tensor<128x512xf32>) attrs =  {iterator_ranges = [128, 512, 256]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):
      %16 = arith.mulf %arg9, %arg10 : f32
      %17 = arith.addf %arg11, %16 : f32
      linalg.yield %17 : f32
  } -> tensor<128x512xf32>

  // Relu
  // CHECK-NEXT: call @xsmm_unary_invoke({{.*}}%[[cast0]], %[[cast0]]
  %c0 = arith.constant 0.0 : f32
  %3 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%2 : tensor<128x512xf32>) {
    ^bb0(%arg9: f32):
      %16 = arith.maxf %arg9, %c0 : f32
      linalg.yield %16 : f32
  } -> tensor<128x512xf32>

  return %3 : tensor<128x512xf32>
}
