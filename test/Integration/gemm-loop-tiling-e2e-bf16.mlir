// RUN: mlir-gen --kernel=args --batch=32 --layers=16,16 --tiles=4,4,4 -float-type=bf16 | tpp-run --M-tile-shape=2 --N-tile-shape=2 --loop-shuffle-order=0,2,1,3 --num-outer-parallel=2  -e=entry -entry-point-result=void -seed 123 -print | FileCheck %s
// RUN: mlir-gen --kernel=args --batch=32 --layers=16,16 --tiles=4,4,4 -float-type=bf16 | tpp-run --def-parallel --parallel-task-grid=2,2  -e=entry -entry-point-result=void -seed 123 -print | FileCheck %s
// CHECK:	( 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.1{{[0-9]+}}, 0.0{{[0-9]+}}, 0.0{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.3{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.3{{[0-9]+}}, 0.0{{[0-9]+}}, 0.3{{[0-9]+}} )
// CHECK:	( 0.4{{[0-9]+}}, 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.1{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.2{{[0-9]+}}, 0.3{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.2{{[0-9]+}}, 0.0{{[0-9]+}}, 0.2{{[0-9]+}} )
// CHECK:	( 0.3{{[0-9]+}}, 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.3{{[0-9]+}}, 0.1{{[0-9]+}}, 0.0{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.3{{[0-9]+}}, 0.2{{[0-9]+}}, 0.0{{[0-9]+}}, 0.3{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.3{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.0{{[0-9]+}}, 0.2{{[0-9]+}} )
// CHECK:	( 0.7{{[0-9]+}}, 0.3{{[0-9]+}}, 0.1{{[0-9]+}}, 0.4{{[0-9]+}} )
// CHECK:	( 0.3{{[0-9]+}}, 0.1{{[0-9]+}}, 0.2{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.2{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.0{{[0-9]+}}, 0.2{{[0-9]+}} )
// CHECK:	( 0.2{{[0-9]+}}, 0.2{{[0-9]+}}, 0.1{{[0-9]+}}, 0.3{{[0-9]+}} )
// CHECK:	( 0.2{{[0-9]+}}, 0.2{{[0-9]+}}, 0.0{{[0-9]+}}, 0.3{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.1{{[0-9]+}}, 0.0{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.1{{[0-9]+}}, 0.4{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.2{{[0-9]+}}, 0.0{{[0-9]+}}, 0.3{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.2{{[0-9]+}}, 0.2{{[0-9]+}}, 0.0{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.1{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.3{{[0-9]+}}, 0.2{{[0-9]+}}, 0.0{{[0-9]+}}, 0.5{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.1{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.0{{[0-9]+}}, 0.0{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.0{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.2{{[0-9]+}}, 0.0{{[0-9]+}}, 0.2{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.1{{[0-9]+}}, 0.2{{[0-9]+}}, 0.2{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.3{{[0-9]+}}, 0.3{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.2{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.2{{[0-9]+}}, 0.0{{[0-9]+}}, 0.4{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.1{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.2{{[0-9]+}}, 0.1{{[0-9]+}}, 0.3{{[0-9]+}}, 0.3{{[0-9]+}} )
// CHECK:	( 0.2{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.0{{[0-9]+}} )
// CHECK:	( 0.2{{[0-9]+}}, 0.5{{[0-9]+}}, 0.0{{[0-9]+}}, 0.2{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.0{{[0-9]+}}, 0.3{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.4{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.2{{[0-9]+}} )
// CHECK:	( 0.3{{[0-9]+}}, 0.0{{[0-9]+}}, 0.2{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.2{{[0-9]+}}, 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.3{{[0-9]+}}, 0.2{{[0-9]+}}, 0.2{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.0{{[0-9]+}}, 0.2{{[0-9]+}}, 0.0{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.3{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.4{{[0-9]+}}, 0.2{{[0-9]+}}, 0.1{{[0-9]+}}, 0.3{{[0-9]+}} )
// CHECK:	( 0.3{{[0-9]+}}, 0.1{{[0-9]+}}, 0.3{{[0-9]+}}, 0.2{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.3{{[0-9]+}}, 0.0{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.2{{[0-9]+}}, 0.2{{[0-9]+}} )
// CHECK:	( 0.2{{[0-9]+}}, 0.1{{[0-9]+}}, 0.1{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.2{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.0{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.2{{[0-9]+}}, 0.1{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.1{{[0-9]+}}, 0.2{{[0-9]+}}, 0.2{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.0{{[0-9]+}}, 0.2{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.2{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.1{{[0-9]+}}, 0.0{{[0-9]+}}, 0.3{{[0-9]+}} )
// CHECK:	( 0.2{{[0-9]+}}, 0.2{{[0-9]+}}, 0.1{{[0-9]+}}, 0.5{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.3{{[0-9]+}}, 0.0{{[0-9]+}}, 0.3{{[0-9]+}} )
// CHECK:	( 0.3{{[0-9]+}}, 0.1{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.4{{[0-9]+}}, 0.2{{[0-9]+}}, 0.1{{[0-9]+}}, 0.5{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.0{{[0-9]+}}, 0.0{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.2{{[0-9]+}}, 0.5{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.1{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.2{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.4{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.3{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.2{{[0-9]+}}, 0.2{{[0-9]+}}, 0.2{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.5{{[0-9]+}}, 0.2{{[0-9]+}}, 0.2{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.1{{[0-9]+}}, 0.3{{[0-9]+}}, 0.0{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.2{{[0-9]+}}, 0.1{{[0-9]+}}, 0.0{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.0{{[0-9]+}}, 0.3{{[0-9]+}}, 0.2{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.3{{[0-9]+}}, 0.1{{[0-9]+}}, 0.0{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.1{{[0-9]+}}, 0.0{{[0-9]+}}, 0.0{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.2{{[0-9]+}}, 0.1{{[0-9]+}}, 0.3{{[0-9]+}} )
// CHECK:	( 0.3{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.2{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.1{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.1{{[0-9]+}}, 0.0{{[0-9]+}}, 0.0{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.1{{[0-9]+}}, 0.1{{[0-9]+}}, 0.4{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.1{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.2{{[0-9]+}} )
// CHECK:	( 0.2{{[0-9]+}}, 0.1{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.2{{[0-9]+}}, 0.0{{[0-9]+}}, 0.2{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.3{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.0{{[0-9]+}}, 0.0{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.2{{[0-9]+}}, 0.0{{[0-9]+}}, 0.3{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.1{{[0-9]+}}, 0.2{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.1{{[0-9]+}}, 0.2{{[0-9]+}}, 0.0{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.2{{[0-9]+}}, 0.2{{[0-9]+}}, 0.0{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.4{{[0-9]+}}, 0.2{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.1{{[0-9]+}}, 0.0{{[0-9]+}}, 0.2{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0, 0.1{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.3{{[0-9]+}} )
// CHECK:	( 0.4{{[0-9]+}}, 0.1{{[0-9]+}}, 0.5{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.2{{[0-9]+}}, 0.3{{[0-9]+}}, 0.1{{[0-9]+}}, 0.2{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.2{{[0-9]+}}, 0.1{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.2{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.2{{[0-9]+}}, 0.3{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.2{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.1{{[0-9]+}}, 0.0{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.1{{[0-9]+}}, 0.1{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.4{{[0-9]+}}, 0.1{{[0-9]+}}, 0.4{{[0-9]+}}, 0.0{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.3{{[0-9]+}}, 0.1{{[0-9]+}}, 0.2{{[0-9]+}} )
// CHECK:	( 0.2{{[0-9]+}}, 0.2{{[0-9]+}}, 0.2{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.0{{[0-9]+}} )
// CHECK:	( 0.3{{[0-9]+}}, 0.4{{[0-9]+}}, 0.1{{[0-9]+}}, 0.4{{[0-9]+}} )
// CHECK:	( 0.3{{[0-9]+}}, 0.1{{[0-9]+}}, 0.1{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.3{{[0-9]+}}, 0.3{{[0-9]+}}, 0.3{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.2{{[0-9]+}}, 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.2{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.2{{[0-9]+}}, 0.0{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.4{{[0-9]+}}, 0.2{{[0-9]+}}, 0.1{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.5{{[0-9]+}}, 0.1{{[0-9]+}}, 0.2{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.0{{[0-9]+}}, 0.0{{[0-9]+}} )
// CHECK:	( 0.2{{[0-9]+}}, 0.2{{[0-9]+}}, 0.2{{[0-9]+}}, 0.2{{[0-9]+}} )
// CHECK:	( 0.2{{[0-9]+}}, 0.4{{[0-9]+}}, 0.2{{[0-9]+}}, 0.4{{[0-9]+}} )
// CHECK:	( 0.2{{[0-9]+}}, 0.1{{[0-9]+}}, 0.0{{[0-9]+}}, 0.5{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.4{{[0-9]+}}, 0.1{{[0-9]+}}, 0.3{{[0-9]+}} )
// CHECK:	( 0.1{{[0-9]+}}, 0.1{{[0-9]+}}, 0.4{{[0-9]+}}, 0.4{{[0-9]+}} )
// CHECK:	( 0.0{{[0-9]+}}, 0.2{{[0-9]+}}, 0.2{{[0-9]+}}, 0.2{{[0-9]+}} )


