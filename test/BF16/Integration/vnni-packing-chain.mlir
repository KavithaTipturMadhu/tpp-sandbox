// RUN: tpp-run %s \
// RUN:  -e entry -entry-point-result=void

func.func @vnni_packing(%arg0: tensor<32x32xbf16>, %arg1: tensor<2x2x8x16x2xbf16>) -> tensor<2x2x8x16x2xbf16> {
  %0 = tensor.empty() : tensor<2x2x16x16xbf16>
  %pack = tensor.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [16, 16]
    into %0 : tensor<32x32xbf16> -> tensor<2x2x16x16xbf16>
  %vnni_pack = tensor.pack %pack inner_dims_pos = [2] inner_tiles = [2]
    into %arg1 : tensor<2x2x16x16xbf16> -> tensor<2x2x8x16x2xbf16>
  return %vnni_pack : tensor<2x2x8x16x2xbf16>
}

func.func @entry() {

  %da = arith.constant dense<[
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0 ],
[32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0 ],
[64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0 ],
[96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0 ],
[16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0 ],
[48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0 ],
[80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0 ],
[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0 ],
[32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0 ],
[64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0 ],
[96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0 ],
[16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0 ],
[48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0 ],
[80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0 ],
[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0 ],
[32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0 ],
[64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0 ],
[96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0 ],
[16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0 ],
[48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0 ],
[80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0 ],
[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0 ],
[32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0 ],
[64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0 ],
[96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0 ],
[16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0 ],
[48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0 ],
[80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0 ],
[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0 ],
[32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0 ],
[64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0 ],
[96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0 ]
  ]> : tensor<32x32xbf16>

  %cst_zero = arith.constant 0.0 : bf16
  %C = tensor.empty() : tensor<2x2x8x16x2xbf16>
  %C_zeroed = linalg.fill ins(%cst_zero : bf16) outs(%C : tensor<2x2x8x16x2xbf16>) -> tensor<2x2x8x16x2xbf16>
  %0 = call @vnni_packing(%da, %C_zeroed) : (tensor<32x32xbf16>, tensor<2x2x8x16x2xbf16>) -> tensor<2x2x8x16x2xbf16>

  %G = arith.constant dense<[
    [ [ [ [ 0.0, 32.0 ], [ 1.0, 33.0 ], [ 2.0, 34.0 ], [ 3.0, 35.0 ], [ 4.0, 36.0 ], [ 5.0, 37.0 ], [ 6.0, 38.0 ], [ 7.0, 39.0 ], [ 8.0, 40.0 ], [ 9.0, 41.0 ], [ 10.0, 42.0 ], [ 11.0, 43.0 ], [ 12.0, 44.0 ], [ 13.0, 45.0 ], [ 14.0, 46.0 ], [ 15.0, 47.0 ] ], [ [ 64.0, 96.0 ], [ 65.0, 97.0 ], [ 66.0, 98.0 ], [ 67.0, 99.0 ], [ 68.0, 100.0 ], [ 69.0, 101.0 ], [ 70.0, 102.0 ], [ 71.0, 103.0 ], [ 72.0, 104.0 ], [ 73.0, 105.0 ], [ 74.0, 106.0 ], [ 75.0, 107.0 ], [ 76.0, 108.0 ], [ 77.0, 109.0 ], [ 78.0, 110.0 ], [ 79.0, 111.0 ] ], [ [ 16.0, 48.0 ], [ 17.0, 49.0 ], [ 18.0, 50.0 ], [ 19.0, 51.0 ], [ 20.0, 52.0 ], [ 21.0, 53.0 ], [ 22.0, 54.0 ], [ 23.0, 55.0 ], [ 24.0, 56.0 ], [ 25.0, 57.0 ], [ 26.0, 58.0 ], [ 27.0, 59.0 ], [ 28.0, 60.0 ], [ 29.0, 61.0 ], [ 30.0, 62.0 ], [ 31.0, 63.0 ] ], [ [ 80.0, 0.0 ], [ 81.0, 1.0 ], [ 82.0, 2.0 ], [ 83.0, 3.0 ], [ 84.0, 4.0 ], [ 85.0, 5.0 ], [ 86.0, 6.0 ], [ 87.0, 7.0 ], [ 88.0, 8.0 ], [ 89.0, 9.0 ], [ 90.0, 10.0 ], [ 91.0, 11.0 ], [ 92.0, 12.0 ], [ 93.0, 13.0 ], [ 94.0, 14.0 ], [ 95.0, 15.0 ] ], [ [ 32.0, 64.0 ], [ 33.0, 65.0 ], [ 34.0, 66.0 ], [ 35.0, 67.0 ], [ 36.0, 68.0 ], [ 37.0, 69.0 ], [ 38.0, 70.0 ], [ 39.0, 71.0 ], [ 40.0, 72.0 ], [ 41.0, 73.0 ], [ 42.0, 74.0 ], [ 43.0, 75.0 ], [ 44.0, 76.0 ], [ 45.0, 77.0 ], [ 46.0, 78.0 ], [ 47.0, 79.0 ] ], [ [ 96.0, 16.0 ], [ 97.0, 17.0 ], [ 98.0, 18.0 ], [ 99.0, 19.0 ], [ 100.0, 20.0 ], [ 101.0, 21.0 ], [ 102.0, 22.0 ], [ 103.0, 23.0 ], [ 104.0, 24.0 ], [ 105.0, 25.0 ], [ 106.0, 26.0 ], [ 107.0, 27.0 ], [ 108.0, 28.0 ], [ 109.0, 29.0 ], [ 110.0, 30.0 ], [ 111.0, 31.0 ] ], [ [ 48.0, 80.0 ], [ 49.0, 81.0 ], [ 50.0, 82.0 ], [ 51.0, 83.0 ], [ 52.0, 84.0 ], [ 53.0, 85.0 ], [ 54.0, 86.0 ], [ 55.0, 87.0 ], [ 56.0, 88.0 ], [ 57.0, 89.0 ], [ 58.0, 90.0 ], [ 59.0, 91.0 ], [ 60.0, 92.0 ], [ 61.0, 93.0 ], [ 62.0, 94.0 ], [ 63.0, 95.0 ] ], [ [ 0.0, 32.0 ], [ 1.0, 33.0 ], [ 2.0, 34.0 ], [ 3.0, 35.0 ], [ 4.0, 36.0 ], [ 5.0, 37.0 ], [ 6.0, 38.0 ], [ 7.0, 39.0 ], [ 8.0, 40.0 ], [ 9.0, 41.0 ], [ 10.0, 42.0 ], [ 11.0, 43.0 ], [ 12.0, 44.0 ], [ 13.0, 45.0 ], [ 14.0, 46.0 ], [ 15.0, 47.0 ] ] ], [ [ [ 64.0, 96.0 ], [ 65.0, 97.0 ], [ 66.0, 98.0 ], [ 67.0, 99.0 ], [ 68.0, 100.0 ], [ 69.0, 101.0 ], [ 70.0, 102.0 ], [ 71.0, 103.0 ], [ 72.0, 104.0 ], [ 73.0, 105.0 ], [ 74.0, 106.0 ], [ 75.0, 107.0 ], [ 76.0, 108.0 ], [ 77.0, 109.0 ], [ 78.0, 110.0 ], [ 79.0, 111.0 ] ], [ [ 16.0, 48.0 ], [ 17.0, 49.0 ], [ 18.0, 50.0 ], [ 19.0, 51.0 ], [ 20.0, 52.0 ], [ 21.0, 53.0 ], [ 22.0, 54.0 ], [ 23.0, 55.0 ], [ 24.0, 56.0 ], [ 25.0, 57.0 ], [ 26.0, 58.0 ], [ 27.0, 59.0 ], [ 28.0, 60.0 ], [ 29.0, 61.0 ], [ 30.0, 62.0 ], [ 31.0, 63.0 ] ], [ [ 80.0, 0.0 ], [ 81.0, 1.0 ], [ 82.0, 2.0 ], [ 83.0, 3.0 ], [ 84.0, 4.0 ], [ 85.0, 5.0 ], [ 86.0, 6.0 ], [ 87.0, 7.0 ], [ 88.0, 8.0 ], [ 89.0, 9.0 ], [ 90.0, 10.0 ], [ 91.0, 11.0 ], [ 92.0, 12.0 ], [ 93.0, 13.0 ], [ 94.0, 14.0 ], [ 95.0, 15.0 ] ], [ [ 32.0, 64.0 ], [ 33.0, 65.0 ], [ 34.0, 66.0 ], [ 35.0, 67.0 ], [ 36.0, 68.0 ], [ 37.0, 69.0 ], [ 38.0, 70.0 ], [ 39.0, 71.0 ], [ 40.0, 72.0 ], [ 41.0, 73.0 ], [ 42.0, 74.0 ], [ 43.0, 75.0 ], [ 44.0, 76.0 ], [ 45.0, 77.0 ], [ 46.0, 78.0 ], [ 47.0, 79.0 ] ], [ [ 96.0, 16.0 ], [ 97.0, 17.0 ], [ 98.0, 18.0 ], [ 99.0, 19.0 ], [ 100.0, 20.0 ], [ 101.0, 21.0 ], [ 102.0, 22.0 ], [ 103.0, 23.0 ], [ 104.0, 24.0 ], [ 105.0, 25.0 ], [ 106.0, 26.0 ], [ 107.0, 27.0 ], [ 108.0, 28.0 ], [ 109.0, 29.0 ], [ 110.0, 30.0 ], [ 111.0, 31.0 ] ], [ [ 48.0, 80.0 ], [ 49.0, 81.0 ], [ 50.0, 82.0 ], [ 51.0, 83.0 ], [ 52.0, 84.0 ], [ 53.0, 85.0 ], [ 54.0, 86.0 ], [ 55.0, 87.0 ], [ 56.0, 88.0 ], [ 57.0, 89.0 ], [ 58.0, 90.0 ], [ 59.0, 91.0 ], [ 60.0, 92.0 ], [ 61.0, 93.0 ], [ 62.0, 94.0 ], [ 63.0, 95.0 ] ], [ [ 0.0, 32.0 ], [ 1.0, 33.0 ], [ 2.0, 34.0 ], [ 3.0, 35.0 ], [ 4.0, 36.0 ], [ 5.0, 37.0 ], [ 6.0, 38.0 ], [ 7.0, 39.0 ], [ 8.0, 40.0 ], [ 9.0, 41.0 ], [ 10.0, 42.0 ], [ 11.0, 43.0 ], [ 12.0, 44.0 ], [ 13.0, 45.0 ], [ 14.0, 46.0 ], [ 15.0, 47.0 ] ], [ [ 64.0, 96.0 ], [ 65.0, 97.0 ], [ 66.0, 98.0 ], [ 67.0, 99.0 ], [ 68.0, 100.0 ], [ 69.0, 101.0 ], [ 70.0, 102.0 ], [ 71.0, 103.0 ], [ 72.0, 104.0 ], [ 73.0, 105.0 ], [ 74.0, 106.0 ], [ 75.0, 107.0 ], [ 76.0, 108.0 ], [ 77.0, 109.0 ], [ 78.0, 110.0 ], [ 79.0, 111.0 ] ] ] ], [ [ [ [ 16.0, 48.0 ], [ 17.0, 49.0 ], [ 18.0, 50.0 ], [ 19.0, 51.0 ], [ 20.0, 52.0 ], [ 21.0, 53.0 ], [ 22.0, 54.0 ], [ 23.0, 55.0 ], [ 24.0, 56.0 ], [ 25.0, 57.0 ], [ 26.0, 58.0 ], [ 27.0, 59.0 ], [ 28.0, 60.0 ], [ 29.0, 61.0 ], [ 30.0, 62.0 ], [ 31.0, 63.0 ] ], [ [ 80.0, 0.0 ], [ 81.0, 1.0 ], [ 82.0, 2.0 ], [ 83.0, 3.0 ], [ 84.0, 4.0 ], [ 85.0, 5.0 ], [ 86.0, 6.0 ], [ 87.0, 7.0 ], [ 88.0, 8.0 ], [ 89.0, 9.0 ], [ 90.0, 10.0 ], [ 91.0, 11.0 ], [ 92.0, 12.0 ], [ 93.0, 13.0 ], [ 94.0, 14.0 ], [ 95.0, 15.0 ] ], [ [ 32.0, 64.0 ], [ 33.0, 65.0 ], [ 34.0, 66.0 ], [ 35.0, 67.0 ], [ 36.0, 68.0 ], [ 37.0, 69.0 ], [ 38.0, 70.0 ], [ 39.0, 71.0 ], [ 40.0, 72.0 ], [ 41.0, 73.0 ], [ 42.0, 74.0 ], [ 43.0, 75.0 ], [ 44.0, 76.0 ], [ 45.0, 77.0 ], [ 46.0, 78.0 ], [ 47.0, 79.0 ] ], [ [ 96.0, 16.0 ], [ 97.0, 17.0 ], [ 98.0, 18.0 ], [ 99.0, 19.0 ], [ 100.0, 20.0 ], [ 101.0, 21.0 ], [ 102.0, 22.0 ], [ 103.0, 23.0 ], [ 104.0, 24.0 ], [ 105.0, 25.0 ], [ 106.0, 26.0 ], [ 107.0, 27.0 ], [ 108.0, 28.0 ], [ 109.0, 29.0 ], [ 110.0, 30.0 ], [ 111.0, 31.0 ] ], [ [ 48.0, 80.0 ], [ 49.0, 81.0 ], [ 50.0, 82.0 ], [ 51.0, 83.0 ], [ 52.0, 84.0 ], [ 53.0, 85.0 ], [ 54.0, 86.0 ], [ 55.0, 87.0 ], [ 56.0, 88.0 ], [ 57.0, 89.0 ], [ 58.0, 90.0 ], [ 59.0, 91.0 ], [ 60.0, 92.0 ], [ 61.0, 93.0 ], [ 62.0, 94.0 ], [ 63.0, 95.0 ] ], [ [ 0.0, 32.0 ], [ 1.0, 33.0 ], [ 2.0, 34.0 ], [ 3.0, 35.0 ], [ 4.0, 36.0 ], [ 5.0, 37.0 ], [ 6.0, 38.0 ], [ 7.0, 39.0 ], [ 8.0, 40.0 ], [ 9.0, 41.0 ], [ 10.0, 42.0 ], [ 11.0, 43.0 ], [ 12.0, 44.0 ], [ 13.0, 45.0 ], [ 14.0, 46.0 ], [ 15.0, 47.0 ] ], [ [ 64.0, 96.0 ], [ 65.0, 97.0 ], [ 66.0, 98.0 ], [ 67.0, 99.0 ], [ 68.0, 100.0 ], [ 69.0, 101.0 ], [ 70.0, 102.0 ], [ 71.0, 103.0 ], [ 72.0, 104.0 ], [ 73.0, 105.0 ], [ 74.0, 106.0 ], [ 75.0, 107.0 ], [ 76.0, 108.0 ], [ 77.0, 109.0 ], [ 78.0, 110.0 ], [ 79.0, 111.0 ] ], [ [ 16.0, 48.0 ], [ 17.0, 49.0 ], [ 18.0, 50.0 ], [ 19.0, 51.0 ], [ 20.0, 52.0 ], [ 21.0, 53.0 ], [ 22.0, 54.0 ], [ 23.0, 55.0 ], [ 24.0, 56.0 ], [ 25.0, 57.0 ], [ 26.0, 58.0 ], [ 27.0, 59.0 ], [ 28.0, 60.0 ], [ 29.0, 61.0 ], [ 30.0, 62.0 ], [ 31.0, 63.0 ] ] ], [ [ [ 80.0, 0.0 ], [ 81.0, 1.0 ], [ 82.0, 2.0 ], [ 83.0, 3.0 ], [ 84.0, 4.0 ], [ 85.0, 5.0 ], [ 86.0, 6.0 ], [ 87.0, 7.0 ], [ 88.0, 8.0 ], [ 89.0, 9.0 ], [ 90.0, 10.0 ], [ 91.0, 11.0 ], [ 92.0, 12.0 ], [ 93.0, 13.0 ], [ 94.0, 14.0 ], [ 95.0, 15.0 ] ], [ [ 32.0, 64.0 ], [ 33.0, 65.0 ], [ 34.0, 66.0 ], [ 35.0, 67.0 ], [ 36.0, 68.0 ], [ 37.0, 69.0 ], [ 38.0, 70.0 ], [ 39.0, 71.0 ], [ 40.0, 72.0 ], [ 41.0, 73.0 ], [ 42.0, 74.0 ], [ 43.0, 75.0 ], [ 44.0, 76.0 ], [ 45.0, 77.0 ], [ 46.0, 78.0 ], [ 47.0, 79.0 ] ], [ [ 96.0, 16.0 ], [ 97.0, 17.0 ], [ 98.0, 18.0 ], [ 99.0, 19.0 ], [ 100.0, 20.0 ], [ 101.0, 21.0 ], [ 102.0, 22.0 ], [ 103.0, 23.0 ], [ 104.0, 24.0 ], [ 105.0, 25.0 ], [ 106.0, 26.0 ], [ 107.0, 27.0 ], [ 108.0, 28.0 ], [ 109.0, 29.0 ], [ 110.0, 30.0 ], [ 111.0, 31.0 ] ], [ [ 48.0, 80.0 ], [ 49.0, 81.0 ], [ 50.0, 82.0 ], [ 51.0, 83.0 ], [ 52.0, 84.0 ], [ 53.0, 85.0 ], [ 54.0, 86.0 ], [ 55.0, 87.0 ], [ 56.0, 88.0 ], [ 57.0, 89.0 ], [ 58.0, 90.0 ], [ 59.0, 91.0 ], [ 60.0, 92.0 ], [ 61.0, 93.0 ], [ 62.0, 94.0 ], [ 63.0, 95.0 ] ], [ [ 0.0, 32.0 ], [ 1.0, 33.0 ], [ 2.0, 34.0 ], [ 3.0, 35.0 ], [ 4.0, 36.0 ], [ 5.0, 37.0 ], [ 6.0, 38.0 ], [ 7.0, 39.0 ], [ 8.0, 40.0 ], [ 9.0, 41.0 ], [ 10.0, 42.0 ], [ 11.0, 43.0 ], [ 12.0, 44.0 ], [ 13.0, 45.0 ], [ 14.0, 46.0 ], [ 15.0, 47.0 ] ], [ [ 64.0, 96.0 ], [ 65.0, 97.0 ], [ 66.0, 98.0 ], [ 67.0, 99.0 ], [ 68.0, 100.0 ], [ 69.0, 101.0 ], [ 70.0, 102.0 ], [ 71.0, 103.0 ], [ 72.0, 104.0 ], [ 73.0, 105.0 ], [ 74.0, 106.0 ], [ 75.0, 107.0 ], [ 76.0, 108.0 ], [ 77.0, 109.0 ], [ 78.0, 110.0 ], [ 79.0, 111.0 ] ], [ [ 16.0, 48.0 ], [ 17.0, 49.0 ], [ 18.0, 50.0 ], [ 19.0, 51.0 ], [ 20.0, 52.0 ], [ 21.0, 53.0 ], [ 22.0, 54.0 ], [ 23.0, 55.0 ], [ 24.0, 56.0 ], [ 25.0, 57.0 ], [ 26.0, 58.0 ], [ 27.0, 59.0 ], [ 28.0, 60.0 ], [ 29.0, 61.0 ], [ 30.0, 62.0 ], [ 31.0, 63.0 ] ], [ [ 80.0, 0.0 ], [ 81.0, 1.0 ], [ 82.0, 2.0 ], [ 83.0, 3.0 ], [ 84.0, 4.0 ], [ 85.0, 5.0 ], [ 86.0, 6.0 ], [ 87.0, 7.0 ], [ 88.0, 8.0 ], [ 89.0, 9.0 ], [ 90.0, 10.0 ], [ 91.0, 11.0 ], [ 92.0, 12.0 ], [ 93.0, 13.0 ], [ 94.0, 14.0 ], [ 95.0, 15.0 ] ] ] ]
  ]> : tensor<2x2x8x16x2xbf16>

  %threshold = arith.constant 0.0 : bf16
  check.expect_almost_eq(%0, %G, %threshold) : tensor<2x2x8x16x2xbf16>, tensor<2x2x8x16x2xbf16>, bf16

  return
}
