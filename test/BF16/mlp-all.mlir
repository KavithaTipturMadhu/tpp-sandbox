// RUN: tpp-opt %s --pack-vnni="block-factors=2"  --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize  --linalg-ext-to-loops --convert-linalg-to-tpp --convert-check-to-loops --convert-vnni-to-tpp --convert-tpp-to-xsmm -convert-xsmm-to-func -convert-vector-to-scf -convert-scf-to-cf  --convert-math-to-llvm --lower-affine --reconcile-unrealized-casts |\
// RUN: tpp-run \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext,%tpplibdir/libtpp_c_runner_utils%shlibext
//
// RUN: tpp-opt %s --pack-vnni="block-factors=2"  --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize  --linalg-ext-to-loops --convert-linalg-to-tpp --convert-vnni-to-tpp | FileCheck %s -check-prefix=TPP
// 
// Total flops = sum(broadcast O(n*m) + matmul O(2*n*m*k) + ReLU (O(n*m))
// 2*128x512 (131072) + 2*128x256x512 (33554432) + 2*128x1024 (262144) + 2*128x512x1024 (134217728) + 2*128x2048 (524288) + 2*128x1024x2048 (536870912) + 2*128x1000 (256000) + 2*128x2048x1000 (524288000) = 1230102376
// BENCH_TOTAL_FLOPS: 1230102376
 
#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// TPP: func.func @entry(
// TPP: %[[ARG0:.+]]: memref<128x256xbf16>
// TPP: %[[ARG1:.+]]: memref<256x512xbf16>,
// TPP: %[[ARG2:.+]]: memref<512xbf16>, 
// TPP: %[[ARG3:.+]]: memref<512x1024xbf16>,
// TPP: %[[ARG4:.+]]: memref<1024xbf16>,
// TPP: %[[ARG5:.+]]: memref<1024x2048xbf16>,
// TPP: %[[ARG6:.+]]: memref<2048xbf16>,
// TPP: %[[ARG7:.+]]: memref<2048x1000xbf16>,
// TPP: %[[ARG8:.+]]: memref<1000xbf16>
// TPP: %[[OUTPUT1:.+]]: memref<128x2048xbf16>,
// TPP: %[[OUTPUT2:.+]]: memref<128x1024xbf16>,
// TPP: %[[OUTPUT3:.+]]: memref<128x512xbf16>,
// TPP: %[[OUTPUT:.+]]: memref<128x1000xbf16>) {
func.func @entry(%arg0: tensor<128x256xbf16>, %arg1: tensor<256x512xbf16>,
                 %arg2: tensor<512xbf16>, %arg3: tensor<512x1024xbf16>,
                 %arg4: tensor<1024xbf16>, %arg5: tensor<1024x2048xbf16>,
                 %arg6: tensor<2048xbf16>, %arg7: tensor<2048x1000xbf16>,
                 %arg8: tensor<1000xbf16>, %output1: tensor<128x2048xbf16>, 
                 %output2: tensor<128x1024xbf16>,%ouput3: tensor<128x512xbf16>, 
		 %output: tensor<128x1000xbf16>) -> tensor<128x1000xbf16> {
  %c0 = arith.constant 0.0 : bf16
// TPP: tpp.identity ins(%[[ARG2]] : memref<512xbf16>) out(%[[OUTPUT3]] : memref<128x512xbf16>)
// TPP: %[[ALLOC0:.+]] = memref.alloc() {alignment = 128 : i64} : memref<128x512x2xbf16>
// TPP: scf.for
// TPP: scf.for
// TPP: scf.for
// TPP:   memref.load
// TPP:   memref.store
// TPP: tpp.vnni_matmul ins(%[[ARG0]] : memref<128x256xbf16>, %[[ALLOC0]] : memref<128x512x2xbf16>) out(%[[OUTPUT3]] : memref<128x512xbf16>)
// TPP: tpp.relu ins(%[[OUTPUT3]] : memref<128x512xbf16>) out(%[[OUTPUT3]] : memref<128x512xbf16>)
  %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<512xbf16>) outs(%ouput3 : tensor<128x512xbf16>) {
  ^bb0(%arg9: bf16, %arg10: bf16):
    linalg.yield %arg9 : bf16
  } -> tensor<128x512xbf16>
  %2 = linalg.matmul ins(%arg0, %arg1: tensor<128x256xbf16>, tensor<256x512xbf16>) outs(%1: tensor<128x512xbf16>) -> tensor<128x512xbf16> 
  %3 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%2 : tensor<128x512xbf16>) {
  ^bb0(%arg9: bf16):
    %16 = arith.maxf %arg9, %c0 : bf16
    linalg.yield %16 : bf16
  } -> tensor<128x512xbf16>
// TPP: tpp.identity ins(%[[ARG4]] : memref<1024xbf16>) out(%[[OUTPUT2]] : memref<128x1024xbf16>)
// TPP: %[[ALLOC1:.+]] = memref.alloc() {alignment = 128 : i64} : memref<256x1024x2xbf16>
// TPP: scf.for
// TPP: scf.for
// TPP: scf.for
// TPP:   memref.load
// TPP:   memref.store
// TPP: tpp.vnni_matmul ins(%[[OUTPUT3]] : memref<128x512xbf16>, %[[ALLOC1]] : memref<256x1024x2xbf16>) out(%[[OUTPUT2]] : memref<128x1024xbf16>)
// TPP: tpp.relu ins(%[[OUTPUT2]] : memref<128x1024xbf16>) out(%[[OUTPUT2]] : memref<128x1024xbf16>)
  %5 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg4 : tensor<1024xbf16>) outs(%output2 : tensor<128x1024xbf16>) {
  ^bb0(%arg9: bf16, %arg10: bf16):
    linalg.yield %arg9 : bf16
  } -> tensor<128x1024xbf16>
  %6 = linalg.matmul  ins(%3, %arg3 : tensor<128x512xbf16>, tensor<512x1024xbf16>) outs(%5 : tensor<128x1024xbf16>) -> tensor<128x1024xbf16>
  %7 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%6 : tensor<128x1024xbf16>)  {
  ^bb0(%arg9: bf16):
    %16 = arith.maxf %arg9, %c0 : bf16
    linalg.yield %16 : bf16
  } -> tensor<128x1024xbf16>

// TPP: tpp.identity ins(%[[ARG6]] : memref<2048xbf16>) out(%[[OUTPUT1]] : memref<128x2048xbf16>)
// TPP: %[[ALLOC2:.+]] = memref.alloc() {alignment = 128 : i64} : memref<512x2048x2xbf16>
// TPP: scf.for
// TPP: scf.for
// TPP: scf.for
// TPP:   memref.load
// TPP:   memref.store
// TPP: tpp.vnni_matmul ins(%[[OUTPUT2]] : memref<128x1024xbf16>, %[[ALLOC2]] : memref<512x2048x2xbf16>) out(%[[OUTPUT1]] : memref<128x2048xbf16>)
// TPP: tpp.relu ins(%[[OUTPUT1]] : memref<128x2048xbf16>) out(%[[OUTPUT1]] : memref<128x2048xbf16>)
  %9 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg6 : tensor<2048xbf16>) outs(%output1 : tensor<128x2048xbf16>) {
  ^bb0(%arg9: bf16, %arg10: bf16):
    linalg.yield %arg9 : bf16
  } -> tensor<128x2048xbf16>
  %10 = linalg.matmul ins(%7, %arg5 : tensor<128x1024xbf16>, tensor<1024x2048xbf16>) outs(%9 : tensor<128x2048xbf16>) -> tensor<128x2048xbf16>
  %11 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%10 : tensor<128x2048xbf16>) {
  ^bb0(%arg9: bf16):
    %16 = arith.maxf %arg9, %c0 : bf16
    linalg.yield %16 : bf16
  } -> tensor<128x2048xbf16>

// TPP: tpp.identity ins(%[[ARG8]] : memref<1000xbf16>) out(%[[OUTPUT]] : memref<128x1000xbf16>)
// TPP: %[[ALLOC3:.+]] = memref.alloc() {alignment = 128 : i64} : memref<1024x1000x2xbf16>
// TPP: scf.for
// TPP: scf.for
// TPP: scf.for
// TPP:   memref.load
// TPP:   memref.store
// TPP: tpp.vnni_matmul ins(%[[OUTPUT1]] : memref<128x2048xbf16>, %[[ALLOC3]] : memref<1024x1000x2xbf16>) out(%[[OUTPUT]] : memref<128x1000xbf16>)
// TPP: tpp.relu ins(%[[OUTPUT]] : memref<128x1000xbf16>) out(%[[OUTPUT]] : memref<128x1000xbf16>)
  %13 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg8 : tensor<1000xbf16>) outs(%output : tensor<128x1000xbf16>) {
  ^bb0(%arg9: bf16, %arg10: bf16):
    linalg.yield %arg9 : bf16
  } -> tensor<128x1000xbf16>
  %14 = linalg.matmul  ins(%11, %arg7 : tensor<128x2048xbf16>, tensor<2048x1000xbf16>) outs(%13 : tensor<128x1000xbf16>) -> tensor<128x1000xbf16>
  %15 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%14 : tensor<128x1000xbf16>) {
  ^bb0(%arg9: bf16):
    %16 = arith.maxf %arg9, %c0 : bf16
    linalg.yield %16 : bf16
  } -> tensor<128x1000xbf16>
   
  %threshold = arith.constant 1.0 : bf16
  %constant = arith.constant 2.74878e+11: bf16
// TPP: %[[ALLOC4:.+]] = memref.alloc() {alignment = 128 : i64} : memref<128x1000xbf16>
// TPP: linalg.fill 
// TPP: check.expect_almost_eq
// TPP: memref.dealloc %[[ALLOC0]] : memref<128x512x2xbf16>
// TPP: memref.dealloc %[[ALLOC1]] : memref<256x1024x2xbf16>
// TPP: memref.dealloc %[[ALLOC2]] : memref<512x2048x2xbf16>
// TPP: memref.dealloc %[[ALLOC3]] : memref<1024x1000x2xbf16>
// TPP: memref.dealloc %[[ALLOC4]] : memref<128x1000xbf16>
  %interim = tensor.empty(): tensor<128x1000xbf16>
  %buf = linalg.fill ins(%constant:bf16) outs(%interim: tensor<128x1000xbf16>) -> tensor<128x1000xbf16>
  check.expect_almost_eq(%15, %buf, %threshold): tensor<128x1000xbf16>, tensor<128x1000xbf16>, bf16
 
  return %15 : tensor<128x1000xbf16>
}
