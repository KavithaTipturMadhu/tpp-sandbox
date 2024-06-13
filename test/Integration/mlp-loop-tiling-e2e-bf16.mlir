// RUN: mlir-gen --kernel=args --bias --relu --batch=32 --layers=16,16,16,16 --tiles=4,4,4 --float-type=bf16| tpp-run --M-tile-shape=2,4 --N-tile-shape=2,2 --loop-shuffle-order=0,2,1,3 --num-outer-parallel=2  -e=entry -entry-point-result=void -seed 123 -print | FileCheck %s
// RUN: mlir-gen --kernel=args --bias --relu --batch=32 --layers=16,16,16,16 --tiles=4,4,4 --float-type=bf16| tpp-run --def-parallel --parallel-task-grid=2,2  -e=entry -entry-point-result=void -seed 123 -print | FileCheck %s
// CHECK:	( 1.1{{[0-9]+}}, 0.2{{[0-9]+}}, 0.8{{[0-9]+}}, 1.0{{[0-9]+}} )
// CHECK:	( 0.8{{[0-9]+}}, 0.2{{[0-9]+}}, 0.6{{[0-9]+}}, 1.1{{[0-9]+}} )
// CHECK:	( 0.6{{[0-9]+}}, 0.1{{[0-9]+}}, 0.5{{[0-9]+}}, 1 )
// CHECK:	( 1.3{{[0-9]+}}, 0.4{{[0-9]+}}, 0.6{{[0-9]+}}, 1.1{{[0-9]+}} )
// CHECK:	( 0.6{{[0-9]+}}, 0.7{{[0-9]+}}, 0.2{{[0-9]+}}, 0.8{{[0-9]+}} )
// CHECK:	( 0.6{{[0-9]+}}, 0.6{{[0-9]+}}, 0.3{{[0-9]+}}, 1.4{{[0-9]+}} )
// CHECK:	( 0.4{{[0-9]+}}, 0.4{{[0-9]+}}, 0.2{{[0-9]+}}, 0.9{{[0-9]+}} )
// CHECK:	( 0.5{{[0-9]+}}, 0.5{{[0-9]+}}, 0.4{{[0-9]+}}, 0.7{{[0-9]+}} )
// CHECK:	( 0.7{{[0-9]+}}, 0.7{{[0-9]+}}, 0.4{{[0-9]+}}, 0.9{{[0-9]+}} )
// CHECK:	( 0.9{{[0-9]+}}, 0.5{{[0-9]+}}, 0.5{{[0-9]+}}, 0.7{{[0-9]+}} )
// CHECK:	( 0.5{{[0-9]+}}, 0.4{{[0-9]+}}, 0.3{{[0-9]+}}, 0.5{{[0-9]+}} )
// CHECK:	( 0.7{{[0-9]+}}, 1.3{{[0-9]+}}, 0.7{{[0-9]+}}, 0.9{{[0-9]+}} )
// CHECK:	( 0.8{{[0-9]+}}, 0.8{{[0-9]+}}, 0.6{{[0-9]+}}, 0.7{{[0-9]+}} )
// CHECK:	( 0.8{{[0-9]+}}, 0.6{{[0-9]+}}, 0.6{{[0-9]+}}, 0.6{{[0-9]+}} )
// CHECK:	( 0.6{{[0-9]+}}, 0.5{{[0-9]+}}, 0.3{{[0-9]+}}, 0.4{{[0-9]+}} )
// CHECK:	( 0.7{{[0-9]+}}, 0.8{{[0-9]+}}, 0.4{{[0-9]+}}, 0.5{{[0-9]+}} )
// CHECK:	( 0.9{{[0-9]+}}, 0.3{{[0-9]+}}, 0.7{{[0-9]+}}, 1.1{{[0-9]+}} )
// CHECK:	( 1.1{{[0-9]+}}, 0.2{{[0-9]+}}, 0.8{{[0-9]+}}, 0.8{{[0-9]+}} )
// CHECK:	( 0.6{{[0-9]+}}, 0.2{{[0-9]+}}, 0.7{{[0-9]+}}, 0.8{{[0-9]+}} )
// CHECK:	( 0.6{{[0-9]+}}, 0.1{{[0-9]+}}, 0.4{{[0-9]+}}, 0.7{{[0-9]+}} )
// CHECK:	( 0.9{{[0-9]+}}, 0.9{{[0-9]+}}, 0.3{{[0-9]+}}, 0.9{{[0-9]+}} )
// CHECK:	( 0.6{{[0-9]+}}, 0.5{{[0-9]+}}, 0.6{{[0-9]+}}, 0.9{{[0-9]+}} )
// CHECK:	( 0.7{{[0-9]+}}, 0.7{{[0-9]+}}, 0.2{{[0-9]+}}, 0.7{{[0-9]+}} )
// CHECK:	( 0.5{{[0-9]+}}, 0.5{{[0-9]+}}, 0.3{{[0-9]+}}, 0.7{{[0-9]+}} )
// CHECK:	( 0.8{{[0-9]+}}, 1.2{{[0-9]+}}, 0.4{{[0-9]+}}, 1.2{{[0-9]+}} )
// CHECK:	( 0.8{{[0-9]+}}, 0.9{{[0-9]+}}, 0.4{{[0-9]+}}, 0.8{{[0-9]+}} )
// CHECK:	( 0.6{{[0-9]+}}, 0.7{{[0-9]+}}, 0.3{{[0-9]+}}, 0.7{{[0-9]+}} )
// CHECK:	( 0.5{{[0-9]+}}, 0.3{{[0-9]+}}, 0.5{{[0-9]+}}, 0.5{{[0-9]+}} )
// CHECK:	( 1.0{{[0-9]+}}, 0.8{{[0-9]+}}, 0.6{{[0-9]+}}, 0.7{{[0-9]+}} )
// CHECK:	( 0.8{{[0-9]+}}, 1.0{{[0-9]+}}, 0.7{{[0-9]+}}, 0.8{{[0-9]+}} )
// CHECK:	( 1.0{{[0-9]+}}, 0.8{{[0-9]+}}, 0.5{{[0-9]+}}, 0.4{{[0-9]+}} )
// CHECK:	( 0.6{{[0-9]+}}, 0.9{{[0-9]+}}, 0.3{{[0-9]+}}, 0.4{{[0-9]+}} )
// CHECK:	( 0.8{{[0-9]+}}, 0.2{{[0-9]+}}, 0.6{{[0-9]+}}, 1.0{{[0-9]+}} )
// CHECK:	( 0.7{{[0-9]+}}, 0.2{{[0-9]+}}, 0.6{{[0-9]+}}, 0.8{{[0-9]+}} )
// CHECK:	( 0.7{{[0-9]+}}, 0.4{{[0-9]+}}, 0.7{{[0-9]+}}, 1.3{{[0-9]+}} )
// CHECK:	( 0.6{{[0-9]+}}, 0.1{{[0-9]+}}, 0.6{{[0-9]+}}, 0.8{{[0-9]+}} )
// CHECK:	( 0.7{{[0-9]+}}, 0.7{{[0-9]+}}, 0.2{{[0-9]+}}, 0.8{{[0-9]+}} )
// CHECK:	( 0.6{{[0-9]+}}, 0.6{{[0-9]+}}, 0.2{{[0-9]+}}, 0.8{{[0-9]+}} )
// CHECK:	( 0.5{{[0-9]+}}, 1.1{{[0-9]+}}, 0.6{{[0-9]+}}, 0.9{{[0-9]+}} )
// CHECK:	( 0.4{{[0-9]+}}, 0.5{{[0-9]+}}, 0.1{{[0-9]+}}, 0.8{{[0-9]+}} )
// CHECK:	( 0.7{{[0-9]+}}, 0.7{{[0-9]+}}, 0.4{{[0-9]+}}, 0.7{{[0-9]+}} )
// CHECK:	( 0.9{{[0-9]+}}, 0.5{{[0-9]+}}, 0.4{{[0-9]+}}, 0.8{{[0-9]+}} )
// CHECK:	( 0.6{{[0-9]+}}, 0.6{{[0-9]+}}, 0.4{{[0-9]+}}, 0.6{{[0-9]+}} )
// CHECK:	( 0.6{{[0-9]+}}, 0.4{{[0-9]+}}, 0.4{{[0-9]+}}, 0.5{{[0-9]+}} )
// CHECK:	( 0.7{{[0-9]+}}, 0.8{{[0-9]+}}, 0.5{{[0-9]+}}, 0.5{{[0-9]+}} )
// CHECK:	( 1.1{{[0-9]+}}, 1.1{{[0-9]+}}, 0.6{{[0-9]+}}, 0.5{{[0-9]+}} )
// CHECK:	( 0.7{{[0-9]+}}, 0.8{{[0-9]+}}, 0.4{{[0-9]+}}, 0.4{{[0-9]+}} )
// CHECK:	( 0.8{{[0-9]+}}, 0.6{{[0-9]+}}, 0.5{{[0-9]+}}, 0.4{{[0-9]+}} )
// CHECK:	( 1.0{{[0-9]+}}, 0.{{[0-9]+}}, 0.8{{[0-9]+}}, 1.3{{[0-9]+}} )
// CHECK:	( 1.0{{[0-9]+}}, 0.5{{[0-9]+}}, 0.5{{[0-9]+}}, 0.8{{[0-9]+}} )
// CHECK:	( 0.8{{[0-9]+}}, 0.2{{[0-9]+}}, 0.7{{[0-9]+}}, 0.9{{[0-9]+}} )
// CHECK:	( 0.8{{[0-9]+}}, 0.2{{[0-9]+}}, 0.7{{[0-9]+}}, 0.9{{[0-9]+}} )
// CHECK:	( 0.7{{[0-9]+}}, 0.7{{[0-9]+}}, 0.3{{[0-9]+}}, 1.0{{[0-9]+}} )
// CHECK:	( 0.8{{[0-9]+}}, 0.6{{[0-9]+}}, 0.1{{[0-9]+}}, 0.6{{[0-9]+}} )
// CHECK:	( 0.6{{[0-9]+}}, 0.9{{[0-9]+}}, 0.3{{[0-9]+}}, 0.9{{[0-9]+}} )
// CHECK:	( 0.7{{[0-9]+}}, 0.9{{[0-9]+}}, 0.4{{[0-9]+}}, 0.9{{[0-9]+}} )
// CHECK:	( 1.0{{[0-9]+}}, 0.7{{[0-9]+}}, 0.6{{[0-9]+}}, 1.2{{[0-9]+}} )
// CHECK:	( 0.7{{[0-9]+}}, 0.6{{[0-9]+}}, 0.4{{[0-9]+}}, 0.7{{[0-9]+}} )
// CHECK:	( 0.8{{[0-9]+}}, 0.9{{[0-9]+}}, 0.5{{[0-9]+}}, 1.0{{[0-9]+}} )
// CHECK:	( 0.8{{[0-9]+}}, 0.8{{[0-9]+}}, 0.4{{[0-9]+}}, 0.8{{[0-9]+}} )
// CHECK:	( 1.3{{[0-9]+}}, 0.9{{[0-9]+}}, 0.7{{[0-9]+}}, 0.5{{[0-9]+}} )
// CHECK:	( 0.6{{[0-9]+}}, 0.5{{[0-9]+}}, 0.5{{[0-9]+}}, 0.7{{[0-9]+}} )
// CHECK:	( 0.8{{[0-9]+}}, 1.1{{[0-9]+}}, 0.8{{[0-9]+}}, 0.5{{[0-9]+}} )
// CHECK:	( 0.8{{[0-9]+}}, 0.8{{[0-9]+}}, 0.5{{[0-9]+}}, 0.6{{[0-9]+}} )
// CHECK:	( 0.7{{[0-9]+}}, 0.1{{[0-9]+}}, 0.6{{[0-9]+}}, 1.0{{[0-9]+}} )
// CHECK:	( 0.9{{[0-9]+}}, 0.4{{[0-9]+}}, 0.8{{[0-9]+}}, 1.0{{[0-9]+}} )
// CHECK:	( 1.2{{[0-9]+}}, 0.5{{[0-9]+}}, 0.6{{[0-9]+}}, 1.2{{[0-9]+}} )
// CHECK:	( 0.9{{[0-9]+}}, 0.2{{[0-9]+}}, 0.6{{[0-9]+}}, 0.8{{[0-9]+}} )
// CHECK:	( 0.5{{[0-9]+}}, 0.8{{[0-9]+}}, 0.2{{[0-9]+}}, 0.7{{[0-9]+}} )
// CHECK:	( 0.6{{[0-9]+}}, 0.9{{[0-9]+}}, 0.2{{[0-9]+}}, 0.9{{[0-9]+}} )
// CHECK:	( 0.7{{[0-9]+}}, 0.9{{[0-9]+}}, 0.4{{[0-9]+}}, 1.1{{[0-9]+}} )
// CHECK:	( 0.7{{[0-9]+}}, 0.4{{[0-9]+}}, 0.2{{[0-9]+}}, 0.7{{[0-9]+}} )
// CHECK:	( 0.9{{[0-9]+}}, 0.7{{[0-9]+}}, 0.{{[0-9]+}}, 0.7{{[0-9]+}} )
// CHECK:	( 0.7{{[0-9]+}}, 0.6{{[0-9]+}}, 0.6{{[0-9]+}}, 1.0{{[0-9]+}} )
// CHECK:	( 1.0{{[0-9]+}}, 0.7{{[0-9]+}}, 0.5{{[0-9]+}}, 0.9{{[0-9]+}} )
// CHECK:	( 0.8{{[0-9]+}}, 0.8{{[0-9]+}}, 1.0{{[0-9]+}}, 0.9{{[0-9]+}} )
// CHECK:	( 0.9{{[0-9]+}}, 0.8{{[0-9]+}}, 0.5{{[0-9]+}}, 0.4{{[0-9]+}} )
// CHECK:	( 0.9{{[0-9]+}}, 0.7{{[0-9]+}}, 0.5{{[0-9]+}}, 0.5{{[0-9]+}} )
// CHECK:	( 1.0{{[0-9]+}}, 1, 0.5{{[0-9]+}}, 0.7{{[0-9]+}} )
// CHECK:	( 0.8{{[0-9]+}}, 1.1{{[0-9]+}}, 0.5{{[0-9]+}}, 0.7{{[0-9]+}} )
// CHECK:	( 0.6{{[0-9]+}}, 0.1{{[0-9]+}}, 0.4{{[0-9]+}}, 0.7{{[0-9]+}} )
// CHECK:	( 0.7{{[0-9]+}}, 0.1{{[0-9]+}}, 0.6{{[0-9]+}}, 0.8{{[0-9]+}} )
// CHECK:	( 1.1{{[0-9]+}}, 0.3{{[0-9]+}}, 0.7{{[0-9]+}}, 0.7{{[0-9]+}} )
// CHECK:	( 0.8{{[0-9]+}}, 0.2{{[0-9]+}}, 0.7{{[0-9]+}}, 0.7{{[0-9]+}} )
// CHECK:	( 0.4{{[0-9]+}}, 0.4{{[0-9]+}}, 0.2{{[0-9]+}}, 0.7{{[0-9]+}} )
// CHECK:	( 0.6{{[0-9]+}}, 0.9{{[0-9]+}}, 0.2{{[0-9]+}}, 0.8{{[0-9]+}} )
// CHECK:	( 0.6{{[0-9]+}}, 0.8{{[0-9]+}}, 0.5{{[0-9]+}}, 0.7{{[0-9]+}} )
// CHECK:	( 0.6{{[0-9]+}}, 0.6{{[0-9]+}}, 0.2{{[0-9]+}}, 0.8{{[0-9]+}} )
// CHECK:	( 0.5{{[0-9]+}}, 0.6{{[0-9]+}}, 0.3{{[0-9]+}}, 0.6{{[0-9]+}} )
// CHECK:	( 0.7{{[0-9]+}}, 0.4{{[0-9]+}}, 0.4{{[0-9]+}}, 1.1{{[0-9]+}} )
// CHECK:	( 0.5{{[0-9]+}}, 0.5{{[0-9]+}}, 0.3{{[0-9]+}}, 0.6{{[0-9]+}} )
// CHECK:	( 0.8{{[0-9]+}}, 1.2{{[0-9]+}}, 0.6{{[0-9]+}}, 1.0{{[0-9]+}} )
// CHECK:	( 0.6{{[0-9]+}}, 0.6{{[0-9]+}}, 0.4{{[0-9]+}}, 0.4{{[0-9]+}} )
// CHECK:	( 0.8{{[0-9]+}}, 0.8{{[0-9]+}}, 0.4{{[0-9]+}}, 0.5{{[0-9]+}} )
// CHECK:	( 1.1{{[0-9]+}}, 0.6{{[0-9]+}}, 0.6{{[0-9]+}}, 0.4{{[0-9]+}} )
// CHECK:	( 0.7{{[0-9]+}}, 0.8{{[0-9]+}}, 0.5{{[0-9]+}}, 0.5{{[0-9]+}} )
// CHECK:	( 0.7{{[0-9]+}}, 0.2{{[0-9]+}}, 0.4{{[0-9]+}}, 1.0{{[0-9]+}} )
// CHECK:	( 1.1{{[0-9]+}}, 0.1{{[0-9]+}}, 0.6{{[0-9]+}}, 0.8{{[0-9]+}} )
// CHECK:	( 1.1{{[0-9]+}}, 0.5{{[0-9]+}}, 1.1{{[0-9]+}}, 1.1{{[0-9]+}} )
// CHECK:	( 0.9{{[0-9]+}}, 0.2{{[0-9]+}}, 0.6{{[0-9]+}}, 1 )
// CHECK:	( 0.7{{[0-9]+}}, 0.6{{[0-9]+}}, 0.1{{[0-9]+}}, 0.8{{[0-9]+}} )
// CHECK:	( 0.6{{[0-9]+}}, 0.6{{[0-9]+}}, 0.3{{[0-9]+}}, 0.8{{[0-9]+}} )
// CHECK:	( 0.8{{[0-9]+}}, 1.0{{[0-9]+}}, 0.4{{[0-9]+}}, 0.9{{[0-9]+}} )
// CHECK:	( 0.7{{[0-9]+}}, 1.0{{[0-9]+}}, 0.3{{[0-9]+}}, 1.0{{[0-9]+}} )
// CHECK:	( 0.6{{[0-9]+}}, 0.5{{[0-9]+}}, 0.3{{[0-9]+}}, 0.6{{[0-9]+}} )
// CHECK:	( 0.6{{[0-9]+}}, 0.5{{[0-9]+}}, 0.3{{[0-9]+}}, 1.1{{[0-9]+}} )
// CHECK:	( 0.9{{[0-9]+}}, 0.9{{[0-9]+}}, 0.5{{[0-9]+}}, 1.5{{[0-9]+}} )
// CHECK:	( 0.8{{[0-9]+}}, {{[0|1]}}{{(\.[0-9]+)?}}, 0.5{{[0-9]+}}, 0.8{{[0-9]+}} )
// CHECK:	( 1.0{{[0-9]+}}, 0.7{{[0-9]+}}, 0.4{{[0-9]+}}, 0.3{{[0-9]+}} )
// CHECK:	( 1.0{{[0-9]+}}, 0.6{{[0-9]+}}, 0.6{{[0-9]+}}, 0.5{{[0-9]+}} )
// CHECK:	( 1.0{{[0-9]+}}, 1.{{[0-9]+}}, 0.5{{[0-9]+}}, 0.6{{[0-9]+}} )
// CHECK:	( 1.2{{[0-9]+}}, 0.9{{[0-9]+}}, 0.6{{[0-9]+}}, 0.7{{[0-9]+}} )
// CHECK:	( 1.5{{[0-9]+}}, 0.2{{[0-9]+}}, 0.8{{[0-9]+}}, 1.4{{[0-9]+}} )
// CHECK:	( 0.8{{[0-9]+}}, 0.1{{[0-9]+}}, 0.5{{[0-9]+}}, 0.9{{[0-9]+}} )
// CHECK:	( 0.9{{[0-9]+}}, 0.3{{[0-9]+}}, 0.7{{[0-9]+}}, 1.1{{[0-9]+}} )
// CHECK:	( 1.0{{[0-9]+}}, 0.4{{[0-9]+}}, 0.6{{[0-9]+}}, 1.0{{[0-9]+}} )
// CHECK:	( 1.2{{[0-9]+}}, 0.7{{[0-9]+}}, 0.5, 1.5{{[0-9]+}} )
// CHECK:	( 0.7{{[0-9]+}}, 0.6{{[0-9]+}}, 0.3{{[0-9]+}}, 0.9{{[0-9]+}} )
// CHECK:	( 0.8{{[0-9]+}}, 0.7{{[0-9]+}}, 0.3{{[0-9]+}}, 1.0{{[0-9]+}} )
// CHECK:	( 0.8{{[0-9]+}}, 0.7{{[0-9]+}}, 0.7{{[0-9]+}}, 1.0{{[0-9]+}} )
// CHECK:	( 1.1{{[0-9]+}}, 0.8{{[0-9]+}}, 0.8{{[0-9]+}}, 1.1{{[0-9]+}} )
// CHECK:	( 0.6{{[0-9]+}}, 0.5{{[0-9]+}}, 0.3{{[0-9]+}}, 0.5{{[0-9]+}} )
// CHECK:	( 0.9{{[0-9]+}}, 0.8{{[0-9]+}}, 0.5{{[0-9]+}}, 0.9{{[0-9]+}} )
// CHECK:	( 1.1{{[0-9]+}}, 1.1{{[0-9]+}}, 0.5{{[0-9]+}}, 0.9{{[0-9]+}} )
// CHECK:	( 1.0{{[0-9]+}}, 0.9{{[0-9]+}}, 1.0{{[0-9]+}}, 0.7{{[0-9]+}} )
// CHECK:	( 0.7{{[0-9]+}}, 0.7{{[0-9]+}}, 0.4{{[0-9]+}}, 0.4{{[0-9]+}} )
// CHECK:	( 0.9{{[0-9]+}}, 0.9{{[0-9]+}}, 0.8{{[0-9]+}}, 0.5{{[0-9]+}} )
// CHECK:	( 1.0{{[0-9]+}}, 0.9{{[0-9]+}}, 0.{{[0-9]+}}, 0.6{{[0-9]+}} )
