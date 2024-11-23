// RUN: tpp-run --linalg-to-vector --insert-transpose %s -e entry --entry-point-result=void -print -seed 123
// RUN: tpp-run %s -e entry --entry-point-result=void -print -seed 123
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4 floordiv 2, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d2)>
module{
func.func @vnni_brgemm_require_transpose_on_C(%arg0: tensor<4x4x4xbf16>, %arg1: tensor<4x2x4x2xbf16>, %arg2:tensor<4x4xbf16>) -> tensor<4x4xbf16> {
  %out = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : tensor<4x4x4xbf16>, tensor<4x2x4x2xbf16>)
    outs(%arg2 : tensor<4x4xbf16>) {
      ^bb0(%in: bf16, %in_5: bf16, %out: bf16):
        %5 = arith.mulf %in, %in_5 : bf16
        %6 = arith.addf %out, %5 : bf16
        linalg.yield %6 : bf16
  }-> tensor<4x4xbf16>
  return %out: tensor<4x4xbf16>
}
}
