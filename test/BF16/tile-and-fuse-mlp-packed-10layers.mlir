#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4 floordiv 2, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @mlp(%arg0: tensor<8x112x32x32xbf16>, %arg1: tensor<112x112x16x32x2xbf16>, %arg2: tensor<3584xbf16>, %arg3: tensor<8x112x32x32xbf16>, %arg8: tensor<112x112x16x32x2xbf16>, %arg9: tensor<3584xbf16>, %arg10: tensor<8x112x32x32xbf16>  , %arg11: tensor<112x112x16x32x2xbf16>, %arg12: tensor<3584xbf16>, %arg13: tensor<8x112x32x32xbf16> , %arg14: tensor<112x112x16x32x2xbf16>, %arg15: tensor<3584xbf16>, %arg16: tensor<8x112x32x32xbf16> , %arg17: tensor<112x112x16x32x2xbf16>, %arg18: tensor<3584xbf16>, %arg19: tensor<8x112x32x32xbf16>  , %arg20: tensor<112x112x16x32x2xbf16>, %arg21: tensor<3584xbf16>, %arg22: tensor<8x112x32x32xbf16>  , %arg23: tensor<112x112x16x32x2xbf16>, %arg24: tensor<3584xbf16>, %arg25: tensor<8x112x32x32xbf16>  , %arg26: tensor<112x112x16x32x2xbf16>, %arg27: tensor<3584xbf16>, %arg28: tensor<8x112x32x32xbf16>  , %arg29: tensor<112x112x16x32x2xbf16>, %arg30: tensor<3584xbf16>, %arg31: tensor<8x112x32x32xbf16>  , %arg32: tensor<112x112x16x32x2xbf16>, %arg33: tensor<3584xbf16>, %arg34: tensor<8x112x32x32xbf16> ) -> tensor<8x112x32x32xbf16> {
    %c112 = arith.constant 112 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant -1.000000e+00 : bf16
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %expanded = tensor.expand_shape %arg2 [[0, 1]] : tensor<3584xbf16> into tensor<112x32xbf16>
    %buf0 = tensor.empty() : tensor<8x112x32x32xbf16>
    %loop0 = scf.for %arg4 = %c0 to %c8 step %c1 iter_args(%arg5 = %buf0) -> (tensor<8x112x32x32xbf16>) {
      %4 = scf.for %arg6 = %c0 to %c112 step %c1 iter_args(%arg7 = %arg5) -> (tensor<8x112x32x32xbf16>) {
        %extracted_slice_1 = tensor.extract_slice %arg0[%arg4, 0, 0, 0] [1, 112, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<112x32x32xbf16>
        %extracted_slice_2 = tensor.extract_slice %arg1[%arg6, 0, 0, 0, 0] [1, 112, 16, 32, 2] [1, 1, 1, 1, 1] : tensor<112x112x16x32x2xbf16> to tensor<112x16x32x2xbf16>
        %extracted_slice_3 = tensor.extract_slice %arg3[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<32x32xbf16>
        %5 = tensor.empty() : tensor<112x16x32x2xbf16>
        %6 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%extracted_slice_1, %extracted_slice_2 : tensor<112x32x32xbf16>, tensor<112x16x32x2xbf16>) outs(%extracted_slice_3 : tensor<32x32xbf16>) {
        ^bb0(%in: bf16, %in_6: bf16, %out: bf16):
          %8 = arith.mulf %in, %in_6 : bf16
          %9 = arith.addf %out, %8 : bf16
          linalg.yield %9 : bf16
        } -> tensor<32x32xbf16>
        %extracted_slice_4 = tensor.extract_slice %expanded[%arg6, 0] [1, 32] [1, 1] : tensor<112x32xbf16> to tensor<32xbf16>
        %extracted_slice_5 = tensor.extract_slice %arg7[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<32x32xbf16>
        %7 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%6, %extracted_slice_4 : tensor<32x32xbf16>, tensor<32xbf16>) outs(%extracted_slice_5 : tensor<32x32xbf16>) {
        ^bb0(%in: bf16, %in_6: bf16, %out: bf16):
          %8 = arith.addf %in, %in_6 : bf16
          %9 = arith.maxf %8, %cst_0 : bf16
          linalg.yield %9 : bf16
        } -> tensor<32x32xbf16>
        %inserted_slice = tensor.insert_slice %7 into %arg7[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xbf16> into tensor<8x112x32x32xbf16>
        scf.yield %inserted_slice : tensor<8x112x32x32xbf16>
      }
      scf.yield %4 : tensor<8x112x32x32xbf16>
    }
  
    %expanded2 = tensor.expand_shape %arg9 [[0, 1]] : tensor<3584xbf16> into tensor<112x32xbf16>
    %buf1 = tensor.empty() : tensor<8x112x32x32xbf16>
    %loop1 = scf.for %arg4 = %c0 to %c8 step %c1 iter_args(%arg5 = %buf1) -> (tensor<8x112x32x32xbf16>) {
      %4 = scf.for %arg6 = %c0 to %c112 step %c1 iter_args(%arg7 = %arg5) -> (tensor<8x112x32x32xbf16>) {
        %extracted_slice_1 = tensor.extract_slice %loop0[%arg4, 0, 0, 0] [1, 112, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<112x32x32xbf16>
        %extracted_slice_2 = tensor.extract_slice %arg8[%arg6, 0, 0, 0, 0] [1, 112, 16, 32, 2] [1, 1, 1, 1, 1] : tensor<112x112x16x32x2xbf16> to tensor<112x16x32x2xbf16>
        %extracted_slice_3 = tensor.extract_slice %arg10[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<32x32xbf16>
        %5 = tensor.empty() : tensor<112x16x32x2xbf16>
        %6 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%extracted_slice_1, %extracted_slice_2 : tensor<112x32x32xbf16>, tensor<112x16x32x2xbf16>) outs(%extracted_slice_3 : tensor<32x32xbf16>) {
        ^bb0(%in: bf16, %in_6: bf16, %out: bf16):
          %8 = arith.mulf %in, %in_6 : bf16
          %9 = arith.addf %out, %8 : bf16
          linalg.yield %9 : bf16
        } -> tensor<32x32xbf16>
        %extracted_slice_4 = tensor.extract_slice %expanded2[%arg6, 0] [1, 32] [1, 1] : tensor<112x32xbf16> to tensor<32xbf16>
        %extracted_slice_5 = tensor.extract_slice %arg7[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<32x32xbf16>
        %7 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%6, %extracted_slice_4 : tensor<32x32xbf16>, tensor<32xbf16>) outs(%extracted_slice_5 : tensor<32x32xbf16>) {
        ^bb0(%in: bf16, %in_6: bf16, %out: bf16):
          %8 = arith.addf %in, %in_6 : bf16
          %9 = arith.maxf %8, %cst_0 : bf16
          linalg.yield %9 : bf16
        } -> tensor<32x32xbf16>
        %inserted_slice = tensor.insert_slice %7 into %arg7[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xbf16> into tensor<8x112x32x32xbf16>
        scf.yield %inserted_slice : tensor<8x112x32x32xbf16>
      }
      scf.yield %4 : tensor<8x112x32x32xbf16>
    }
    %expanded3 = tensor.expand_shape %arg12 [[0, 1]] : tensor<3584xbf16> into tensor<112x32xbf16>
    %buf2 = tensor.empty() : tensor<8x112x32x32xbf16>
    %loop2 = scf.for %arg4 = %c0 to %c8 step %c1 iter_args(%arg5 = %buf2) -> (tensor<8x112x32x32xbf16>) {
      %4 = scf.for %arg6 = %c0 to %c112 step %c1 iter_args(%arg7 = %arg5) -> (tensor<8x112x32x32xbf16>) {
        %extracted_slice_1 = tensor.extract_slice %loop1[%arg4, 0, 0, 0] [1, 112, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<112x32x32xbf16>
        %extracted_slice_2 = tensor.extract_slice %arg11[%arg6, 0, 0, 0, 0] [1, 112, 16, 32, 2] [1, 1, 1, 1, 1] : tensor<112x112x16x32x2xbf16> to tensor<112x16x32x2xbf16>
        %extracted_slice_3 = tensor.extract_slice %arg13[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<32x32xbf16>
        %5 = tensor.empty() : tensor<112x16x32x2xbf16>
        %6 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%extracted_slice_1, %extracted_slice_2 : tensor<112x32x32xbf16>, tensor<112x16x32x2xbf16>) outs(%extracted_slice_3 : tensor<32x32xbf16>) {
        ^bb0(%in: bf16, %in_6: bf16, %out: bf16):
          %8 = arith.mulf %in, %in_6 : bf16
          %9 = arith.addf %out, %8 : bf16
          linalg.yield %9 : bf16
        } -> tensor<32x32xbf16>
        %extracted_slice_4 = tensor.extract_slice %expanded3[%arg6, 0] [1, 32] [1, 1] : tensor<112x32xbf16> to tensor<32xbf16>
        %extracted_slice_5 = tensor.extract_slice %arg7[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<32x32xbf16>
        %7 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%6, %extracted_slice_4 : tensor<32x32xbf16>, tensor<32xbf16>) outs(%extracted_slice_5 : tensor<32x32xbf16>) {
        ^bb0(%in: bf16, %in_6: bf16, %out: bf16):
          %8 = arith.addf %in, %in_6 : bf16
          %9 = arith.maxf %8, %cst_0 : bf16
          linalg.yield %9 : bf16
        } -> tensor<32x32xbf16>
        %inserted_slice = tensor.insert_slice %7 into %arg7[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xbf16> into tensor<8x112x32x32xbf16>
        scf.yield %inserted_slice : tensor<8x112x32x32xbf16>
      }
      scf.yield %4 : tensor<8x112x32x32xbf16>
    }

    %expanded4 = tensor.expand_shape %arg15 [[0, 1]] : tensor<3584xbf16> into tensor<112x32xbf16>
    %buf3 = tensor.empty() : tensor<8x112x32x32xbf16>
    %loop3 = scf.for %arg4 = %c0 to %c8 step %c1 iter_args(%arg5 = %buf3) -> (tensor<8x112x32x32xbf16>) {
      %4 = scf.for %arg6 = %c0 to %c112 step %c1 iter_args(%arg7 = %arg5) -> (tensor<8x112x32x32xbf16>) {
        %extracted_slice_1 = tensor.extract_slice %loop2[%arg4, 0, 0, 0] [1, 112, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<112x32x32xbf16>
        %extracted_slice_2 = tensor.extract_slice %arg14[%arg6, 0, 0, 0, 0] [1, 112, 16, 32, 2] [1, 1, 1, 1, 1] : tensor<112x112x16x32x2xbf16> to tensor<112x16x32x2xbf16>
        %extracted_slice_3 = tensor.extract_slice %arg16[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<32x32xbf16>
        %5 = tensor.empty() : tensor<112x16x32x2xbf16>
        %6 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%extracted_slice_1, %extracted_slice_2 : tensor<112x32x32xbf16>, tensor<112x16x32x2xbf16>) outs(%extracted_slice_3 : tensor<32x32xbf16>) {
        ^bb0(%in: bf16, %in_6: bf16, %out: bf16):
          %8 = arith.mulf %in, %in_6 : bf16
          %9 = arith.addf %out, %8 : bf16
          linalg.yield %9 : bf16
        } -> tensor<32x32xbf16>
        %extracted_slice_4 = tensor.extract_slice %expanded4[%arg6, 0] [1, 32] [1, 1] : tensor<112x32xbf16> to tensor<32xbf16>
        %extracted_slice_5 = tensor.extract_slice %arg7[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<32x32xbf16>
        %7 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%6, %extracted_slice_4 : tensor<32x32xbf16>, tensor<32xbf16>) outs(%extracted_slice_5 : tensor<32x32xbf16>) {
        ^bb0(%in: bf16, %in_6: bf16, %out: bf16):
          %8 = arith.addf %in, %in_6 : bf16
          %9 = arith.maxf %8, %cst_0 : bf16
          linalg.yield %9 : bf16
        } -> tensor<32x32xbf16>
        %inserted_slice = tensor.insert_slice %7 into %arg7[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xbf16> into tensor<8x112x32x32xbf16>
        scf.yield %inserted_slice : tensor<8x112x32x32xbf16>
      }
      scf.yield %4 : tensor<8x112x32x32xbf16>
    }

    %expanded5 = tensor.expand_shape %arg18 [[0, 1]] : tensor<3584xbf16> into tensor<112x32xbf16>
    %buf4 = tensor.empty() : tensor<8x112x32x32xbf16>
    %loop4 = scf.for %arg4 = %c0 to %c8 step %c1 iter_args(%arg5 = %buf4) -> (tensor<8x112x32x32xbf16>) {
      %4 = scf.for %arg6 = %c0 to %c112 step %c1 iter_args(%arg7 = %arg5) -> (tensor<8x112x32x32xbf16>) {
        %extracted_slice_1 = tensor.extract_slice %loop3[%arg4, 0, 0, 0] [1, 112, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<112x32x32xbf16>
        %extracted_slice_2 = tensor.extract_slice %arg17[%arg6, 0, 0, 0, 0] [1, 112, 16, 32, 2] [1, 1, 1, 1, 1] : tensor<112x112x16x32x2xbf16> to tensor<112x16x32x2xbf16>
        %extracted_slice_3 = tensor.extract_slice %arg19[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<32x32xbf16>
        %5 = tensor.empty() : tensor<112x16x32x2xbf16>
        %6 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%extracted_slice_1, %extracted_slice_2 : tensor<112x32x32xbf16>, tensor<112x16x32x2xbf16>) outs(%extracted_slice_3 : tensor<32x32xbf16>) {
        ^bb0(%in: bf16, %in_6: bf16, %out: bf16):
          %8 = arith.mulf %in, %in_6 : bf16
          %9 = arith.addf %out, %8 : bf16
          linalg.yield %9 : bf16
        } -> tensor<32x32xbf16>
        %extracted_slice_4 = tensor.extract_slice %expanded5[%arg6, 0] [1, 32] [1, 1] : tensor<112x32xbf16> to tensor<32xbf16>
        %extracted_slice_5 = tensor.extract_slice %arg7[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<32x32xbf16>
        %7 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%6, %extracted_slice_4 : tensor<32x32xbf16>, tensor<32xbf16>) outs(%extracted_slice_5 : tensor<32x32xbf16>) {
        ^bb0(%in: bf16, %in_6: bf16, %out: bf16):
          %8 = arith.addf %in, %in_6 : bf16
          %9 = arith.maxf %8, %cst_0 : bf16
          linalg.yield %9 : bf16
        } -> tensor<32x32xbf16>
        %inserted_slice = tensor.insert_slice %7 into %arg7[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xbf16> into tensor<8x112x32x32xbf16>
        scf.yield %inserted_slice : tensor<8x112x32x32xbf16>
      }
      scf.yield %4 : tensor<8x112x32x32xbf16>
    }

    %expanded6 = tensor.expand_shape %arg21 [[0, 1]] : tensor<3584xbf16> into tensor<112x32xbf16>
    %buf5 = tensor.empty() : tensor<8x112x32x32xbf16>
    %loop5 = scf.for %arg4 = %c0 to %c8 step %c1 iter_args(%arg5 = %buf5) -> (tensor<8x112x32x32xbf16>) {
      %4 = scf.for %arg6 = %c0 to %c112 step %c1 iter_args(%arg7 = %arg5) -> (tensor<8x112x32x32xbf16>) {
        %extracted_slice_1 = tensor.extract_slice %loop4[%arg4, 0, 0, 0] [1, 112, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<112x32x32xbf16>
        %extracted_slice_2 = tensor.extract_slice %arg20[%arg6, 0, 0, 0, 0] [1, 112, 16, 32, 2] [1, 1, 1, 1, 1] : tensor<112x112x16x32x2xbf16> to tensor<112x16x32x2xbf16>
        %extracted_slice_3 = tensor.extract_slice %arg22[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<32x32xbf16>
        %5 = tensor.empty() : tensor<112x16x32x2xbf16>
        %6 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%extracted_slice_1, %extracted_slice_2 : tensor<112x32x32xbf16>, tensor<112x16x32x2xbf16>) outs(%extracted_slice_3 : tensor<32x32xbf16>) {
        ^bb0(%in: bf16, %in_6: bf16, %out: bf16):
          %8 = arith.mulf %in, %in_6 : bf16
          %9 = arith.addf %out, %8 : bf16
          linalg.yield %9 : bf16
        } -> tensor<32x32xbf16>
        %extracted_slice_4 = tensor.extract_slice %expanded6[%arg6, 0] [1, 32] [1, 1] : tensor<112x32xbf16> to tensor<32xbf16>
        %extracted_slice_5 = tensor.extract_slice %arg7[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<32x32xbf16>
        %7 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%6, %extracted_slice_4 : tensor<32x32xbf16>, tensor<32xbf16>) outs(%extracted_slice_5 : tensor<32x32xbf16>) {
        ^bb0(%in: bf16, %in_6: bf16, %out: bf16):
          %8 = arith.addf %in, %in_6 : bf16
          %9 = arith.maxf %8, %cst_0 : bf16
          linalg.yield %9 : bf16
        } -> tensor<32x32xbf16>
        %inserted_slice = tensor.insert_slice %7 into %arg7[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xbf16> into tensor<8x112x32x32xbf16>
        scf.yield %inserted_slice : tensor<8x112x32x32xbf16>
      }
      scf.yield %4 : tensor<8x112x32x32xbf16>
    }

    %expanded7 = tensor.expand_shape %arg24 [[0, 1]] : tensor<3584xbf16> into tensor<112x32xbf16>
    %buf6 = tensor.empty() : tensor<8x112x32x32xbf16>
    %loop6 = scf.for %arg4 = %c0 to %c8 step %c1 iter_args(%arg5 = %buf6) -> (tensor<8x112x32x32xbf16>) {
      %4 = scf.for %arg6 = %c0 to %c112 step %c1 iter_args(%arg7 = %arg5) -> (tensor<8x112x32x32xbf16>) {
        %extracted_slice_1 = tensor.extract_slice %loop5[%arg4, 0, 0, 0] [1, 112, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<112x32x32xbf16>
        %extracted_slice_2 = tensor.extract_slice %arg23[%arg6, 0, 0, 0, 0] [1, 112, 16, 32, 2] [1, 1, 1, 1, 1] : tensor<112x112x16x32x2xbf16> to tensor<112x16x32x2xbf16>
        %extracted_slice_3 = tensor.extract_slice %arg25[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<32x32xbf16>
        %5 = tensor.empty() : tensor<112x16x32x2xbf16>
        %6 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%extracted_slice_1, %extracted_slice_2 : tensor<112x32x32xbf16>, tensor<112x16x32x2xbf16>) outs(%extracted_slice_3 : tensor<32x32xbf16>) {
        ^bb0(%in: bf16, %in_6: bf16, %out: bf16):
          %8 = arith.mulf %in, %in_6 : bf16
          %9 = arith.addf %out, %8 : bf16
          linalg.yield %9 : bf16
        } -> tensor<32x32xbf16>
        %extracted_slice_4 = tensor.extract_slice %expanded7[%arg6, 0] [1, 32] [1, 1] : tensor<112x32xbf16> to tensor<32xbf16>
        %extracted_slice_5 = tensor.extract_slice %arg7[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<32x32xbf16>
        %7 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%6, %extracted_slice_4 : tensor<32x32xbf16>, tensor<32xbf16>) outs(%extracted_slice_5 : tensor<32x32xbf16>) {
        ^bb0(%in: bf16, %in_6: bf16, %out: bf16):
          %8 = arith.addf %in, %in_6 : bf16
          %9 = arith.maxf %8, %cst_0 : bf16
          linalg.yield %9 : bf16
        } -> tensor<32x32xbf16>
        %inserted_slice = tensor.insert_slice %7 into %arg7[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xbf16> into tensor<8x112x32x32xbf16>
        scf.yield %inserted_slice : tensor<8x112x32x32xbf16>
      }
      scf.yield %4 : tensor<8x112x32x32xbf16>
    }

    %expanded8 = tensor.expand_shape %arg27 [[0, 1]] : tensor<3584xbf16> into tensor<112x32xbf16>
    %buf7 = tensor.empty() : tensor<8x112x32x32xbf16>
    %loop7 = scf.for %arg4 = %c0 to %c8 step %c1 iter_args(%arg5 = %buf7) -> (tensor<8x112x32x32xbf16>) {
      %4 = scf.for %arg6 = %c0 to %c112 step %c1 iter_args(%arg7 = %arg5) -> (tensor<8x112x32x32xbf16>) {
        %extracted_slice_1 = tensor.extract_slice %loop6[%arg4, 0, 0, 0] [1, 112, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<112x32x32xbf16>
        %extracted_slice_2 = tensor.extract_slice %arg26[%arg6, 0, 0, 0, 0] [1, 112, 16, 32, 2] [1, 1, 1, 1, 1] : tensor<112x112x16x32x2xbf16> to tensor<112x16x32x2xbf16>
        %extracted_slice_3 = tensor.extract_slice %arg28[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<32x32xbf16>
        %5 = tensor.empty() : tensor<112x16x32x2xbf16>
        %6 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%extracted_slice_1, %extracted_slice_2 : tensor<112x32x32xbf16>, tensor<112x16x32x2xbf16>) outs(%extracted_slice_3 : tensor<32x32xbf16>) {
        ^bb0(%in: bf16, %in_6: bf16, %out: bf16):
          %8 = arith.mulf %in, %in_6 : bf16
          %9 = arith.addf %out, %8 : bf16
          linalg.yield %9 : bf16
        } -> tensor<32x32xbf16>
        %extracted_slice_4 = tensor.extract_slice %expanded8[%arg6, 0] [1, 32] [1, 1] : tensor<112x32xbf16> to tensor<32xbf16>
        %extracted_slice_5 = tensor.extract_slice %arg7[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<32x32xbf16>
        %7 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%6, %extracted_slice_4 : tensor<32x32xbf16>, tensor<32xbf16>) outs(%extracted_slice_5 : tensor<32x32xbf16>) {
        ^bb0(%in: bf16, %in_6: bf16, %out: bf16):
          %8 = arith.addf %in, %in_6 : bf16
          %9 = arith.maxf %8, %cst_0 : bf16
          linalg.yield %9 : bf16
        } -> tensor<32x32xbf16>
        %inserted_slice = tensor.insert_slice %7 into %arg7[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xbf16> into tensor<8x112x32x32xbf16>
        scf.yield %inserted_slice : tensor<8x112x32x32xbf16>
      }
      scf.yield %4 : tensor<8x112x32x32xbf16>
    }

    %expanded9 = tensor.expand_shape %arg30 [[0, 1]] : tensor<3584xbf16> into tensor<112x32xbf16>
    %buf8 = tensor.empty() : tensor<8x112x32x32xbf16>
    %loop8 = scf.for %arg4 = %c0 to %c8 step %c1 iter_args(%arg5 = %buf8) -> (tensor<8x112x32x32xbf16>) {
      %4 = scf.for %arg6 = %c0 to %c112 step %c1 iter_args(%arg7 = %arg5) -> (tensor<8x112x32x32xbf16>) {
        %extracted_slice_1 = tensor.extract_slice %loop7[%arg4, 0, 0, 0] [1, 112, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<112x32x32xbf16>
        %extracted_slice_2 = tensor.extract_slice %arg29[%arg6, 0, 0, 0, 0] [1, 112, 16, 32, 2] [1, 1, 1, 1, 1] : tensor<112x112x16x32x2xbf16> to tensor<112x16x32x2xbf16>
        %extracted_slice_3 = tensor.extract_slice %arg31[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<32x32xbf16>
        %5 = tensor.empty() : tensor<112x16x32x2xbf16>
        %6 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%extracted_slice_1, %extracted_slice_2 : tensor<112x32x32xbf16>, tensor<112x16x32x2xbf16>) outs(%extracted_slice_3 : tensor<32x32xbf16>) {
        ^bb0(%in: bf16, %in_6: bf16, %out: bf16):
          %8 = arith.mulf %in, %in_6 : bf16
          %9 = arith.addf %out, %8 : bf16
          linalg.yield %9 : bf16
        } -> tensor<32x32xbf16>
        %extracted_slice_4 = tensor.extract_slice %expanded9[%arg6, 0] [1, 32] [1, 1] : tensor<112x32xbf16> to tensor<32xbf16>
        %extracted_slice_5 = tensor.extract_slice %arg7[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<32x32xbf16>
        %7 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%6, %extracted_slice_4 : tensor<32x32xbf16>, tensor<32xbf16>) outs(%extracted_slice_5 : tensor<32x32xbf16>) {
        ^bb0(%in: bf16, %in_6: bf16, %out: bf16):
          %8 = arith.addf %in, %in_6 : bf16
          %9 = arith.maxf %8, %cst_0 : bf16
          linalg.yield %9 : bf16
        } -> tensor<32x32xbf16>
        %inserted_slice = tensor.insert_slice %7 into %arg7[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xbf16> into tensor<8x112x32x32xbf16>
        scf.yield %inserted_slice : tensor<8x112x32x32xbf16>
      }
      scf.yield %4 : tensor<8x112x32x32xbf16>
    }

    %expanded10 = tensor.expand_shape %arg33 [[0, 1]] : tensor<3584xbf16> into tensor<112x32xbf16>
    %buf9 = tensor.empty() : tensor<8x112x32x32xbf16>
    %loop9 = scf.for %arg4 = %c0 to %c8 step %c1 iter_args(%arg5 = %buf9) -> (tensor<8x112x32x32xbf16>) {
      %4 = scf.for %arg6 = %c0 to %c112 step %c1 iter_args(%arg7 = %arg5) -> (tensor<8x112x32x32xbf16>) {
        %extracted_slice_1 = tensor.extract_slice %loop8[%arg4, 0, 0, 0] [1, 112, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<112x32x32xbf16>
        %extracted_slice_2 = tensor.extract_slice %arg32[%arg6, 0, 0, 0, 0] [1, 112, 16, 32, 2] [1, 1, 1, 1, 1] : tensor<112x112x16x32x2xbf16> to tensor<112x16x32x2xbf16>
        %extracted_slice_3 = tensor.extract_slice %arg34[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<32x32xbf16>
        %5 = tensor.empty() : tensor<112x16x32x2xbf16>
        %6 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%extracted_slice_1, %extracted_slice_2 : tensor<112x32x32xbf16>, tensor<112x16x32x2xbf16>) outs(%extracted_slice_3 : tensor<32x32xbf16>) {
        ^bb0(%in: bf16, %in_6: bf16, %out: bf16):
          %8 = arith.mulf %in, %in_6 : bf16
          %9 = arith.addf %out, %8 : bf16
          linalg.yield %9 : bf16
        } -> tensor<32x32xbf16>
        %extracted_slice_4 = tensor.extract_slice %expanded10[%arg6, 0] [1, 32] [1, 1] : tensor<112x32xbf16> to tensor<32xbf16>
        %extracted_slice_5 = tensor.extract_slice %arg7[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<32x32xbf16>
        %7 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%6, %extracted_slice_4 : tensor<32x32xbf16>, tensor<32xbf16>) outs(%extracted_slice_5 : tensor<32x32xbf16>) {
        ^bb0(%in: bf16, %in_6: bf16, %out: bf16):
          %8 = arith.addf %in, %in_6 : bf16
          %9 = arith.maxf %8, %cst_0 : bf16
          linalg.yield %9 : bf16
        } -> tensor<32x32xbf16>
        %inserted_slice = tensor.insert_slice %7 into %arg7[%arg4, %arg6, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xbf16> into tensor<8x112x32x32xbf16>
        scf.yield %inserted_slice : tensor<8x112x32x32xbf16>
      }
      scf.yield %4 : tensor<8x112x32x32xbf16>
    }

    %extracted_slice = tensor.extract_slice %loop9[%c0, %c0, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<4x4xbf16, strided<[4, 1], offset: ?>>
    %2 = vector.transfer_read %extracted_slice[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<4x4xbf16, strided<[4, 1], offset: ?>>, vector<4x4xbf16>
    %3 = arith.extf %2 : vector<4x4xbf16> to vector<4x4xf32>
    vector.print %3 : vector<4x4xf32>
    return %loop9 : tensor<8x112x32x32xbf16>
  }
}

