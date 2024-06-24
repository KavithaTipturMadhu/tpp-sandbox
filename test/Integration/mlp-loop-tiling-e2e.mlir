// RUN: mlir-gen --kernel=args --bias --relu --batch=32 --layers=16,16,16,16 --tiles=4,4,4 | tpp-run --M-tile-shape=2 --N-tile-shape=2 --loop-shuffle-order=0,2,1,3 --num-outer-parallel=2  -e=entry -entry-point-result=void -seed 123 -print | FileCheck %s
// RUN: mlir-gen --kernel=args --bias --relu --batch=32 --layers=16,16,16,16 --tiles=4,4,4 | tpp-run --linalg-to-loops -e=entry -entry-point-result=void -seed 123 -print | FileCheck %s
// RUN: mlir-gen --kernel=args --bias --relu --batch=32 --layers=16,16,16,16 --tiles=4,4,4 | tpp-run --def-parallel   -e=entry -entry-point-result=void -seed 123 -print | FileCheck %s
//	CHECK:	( 1.16{{[0-9]+}}, 0.22{{[0-9]+}}, 0.87{{[0-9]+}}, 1.05{{[0-9]+}} )
//	CHECK:	( 0.81{{[0-9]+}}, 0.26{{[0-9]+}}, 0.64{{[0-9]+}}, 1.19{{[0-9]+}} )
//	CHECK:	( 0.61{{[0-9]+}}, 0.15{{[0-9]+}}, 0.58{{[0-9]+}}, 1.00{{[0-9]+}} )
//	CHECK:	( 1.31{{[0-9]+}}, 0.42{{[0-9]+}}, 0.66{{[0-9]+}}, 1.12{{[0-9]+}} )
//	CHECK:	( 0.64{{[0-9]+}}, 0.75{{[0-9]+}}, 0.29{{[0-9]+}}, 0.88{{[0-9]+}} )
//	CHECK:	( 0.65{{[0-9]+}}, 0.61{{[0-9]+}}, 0.30{{[0-9]+}}, 1.46{{[0-9]+}} )
//	CHECK:	( 0.44{{[0-9]+}}, 0.42{{[0-9]+}}, 0.29{{[0-9]+}}, 0.92{{[0-9]+}} )
//	CHECK:	( 0.57{{[0-9]+}}, 0.53{{[0-9]+}}, 0.46{{[0-9]+}}, 0.74{{[0-9]+}} )
//	CHECK:	( 0.76{{[0-9]+}}, 0.70{{[0-9]+}}, 0.45{{[0-9]+}}, 0.97{{[0-9]+}} )
//	CHECK:	( 0.98{{[0-9]+}}, 0.55{{[0-9]+}}, 0.56{{[0-9]+}}, 0.70{{[0-9]+}} )
//	CHECK:	( 0.53{{[0-9]+}}, 0.46{{[0-9]+}}, 0.39{{[0-9]+}}, 0.57{{[0-9]+}} )
//	CHECK:	( 0.72{{[0-9]+}}, 1.30{{[0-9]+}}, 0.73{{[0-9]+}}, 0.93{{[0-9]+}} )
//	CHECK:	( 0.89{{[0-9]+}}, 0.80{{[0-9]+}}, 0.61{{[0-9]+}}, 0.74{{[0-9]+}} )
//	CHECK:	( 0.89{{[0-9]+}}, 0.68{{[0-9]+}}, 0.64{{[0-9]+}}, 0.63{{[0-9]+}} )
//	CHECK:	( 0.63{{[0-9]+}}, 0.57{{[0-9]+}}, 0.38{{[0-9]+}}, 0.44{{[0-9]+}} )
//	CHECK:	( 0.76{{[0-9]+}}, 0.80{{[0-9]+}}, 0.49{{[0-9]+}}, 0.54{{[0-9]+}} )
//	CHECK:	( 0.95{{[0-9]+}}, 0.33{{[0-9]+}}, 0.75{{[0-9]+}}, 1.18{{[0-9]+}} )
//	CHECK:	( 1.12{{[0-9]+}}, 0.26{{[0-9]+}}, 0.84{{[0-9]+}}, 0.88{{[0-9]+}} )
//	CHECK:	( 0.67{{[0-9]+}}, 0.20{{[0-9]+}}, 0.76{{[0-9]+}}, 0.82{{[0-9]+}} )
//	CHECK:	( 0.61{{[0-9]+}}, 0.15{{[0-9]+}}, 0.46{{[0-9]+}}, 0.79{{[0-9]+}} )
//	CHECK:	( 0.91{{[0-9]+}}, 0.92{{[0-9]+}}, 0.39{{[0-9]+}}, 0.93{{[0-9]+}} )
//	CHECK:	( 0.67{{[0-9]+}}, 0.59{{[0-9]+}}, 0.60{{[0-9]+}}, 0.96{{[0-9]+}} )
//	CHECK:	( 0.70{{[0-9]+}}, 0.74{{[0-9]+}}, 0.23{{[0-9]+}}, 0.75{{[0-9]+}} )
//	CHECK:	( 0.59{{[0-9]+}}, 0.57{{[0-9]+}}, 0.31{{[0-9]+}}, 0.70{{[0-9]+}} )
//	CHECK:	( 0.83{{[0-9]+}}, 1.26{{[0-9]+}}, 0.41{{[0-9]+}}, 1.23{{[0-9]+}} )
//	CHECK:	( 0.82{{[0-9]+}}, 0.95{{[0-9]+}}, 0.43{{[0-9]+}}, 0.82{{[0-9]+}} )
//	CHECK:	( 0.62{{[0-9]+}}, 0.71{{[0-9]+}}, 0.34{{[0-9]+}}, 0.79{{[0-9]+}} )
//	CHECK:	( 0.52{{[0-9]+}}, 0.39{{[0-9]+}}, 0.56{{[0-9]+}}, 0.56{{[0-9]+}} )
//	CHECK:	( 1.02{{[0-9]+}}, 0.81{{[0-9]+}}, 0.67{{[0-9]+}}, 0.78{{[0-9]+}} )
//	CHECK:	( 0.89{{[0-9]+}}, 1.02{{[0-9]+}}, 0.78{{[0-9]+}}, 0.87{{[0-9]+}} )
//	CHECK:	( 1.02{{[0-9]+}}, 0.86{{[0-9]+}}, 0.50{{[0-9]+}}, 0.46{{[0-9]+}} )
//	CHECK:	( 0.67{{[0-9]+}}, 0.98{{[0-9]+}}, 0.36{{[0-9]+}}, 0.40{{[0-9]+}} )
//	CHECK:	( 0.80{{[0-9]+}}, 0.23{{[0-9]+}}, 0.62{{[0-9]+}}, 1.09{{[0-9]+}} )
//	CHECK:	( 0.76{{[0-9]+}}, 0.22{{[0-9]+}}, 0.62{{[0-9]+}}, 0.83{{[0-9]+}} )
//	CHECK:	( 0.78{{[0-9]+}}, 0.42{{[0-9]+}}, 0.70{{[0-9]+}}, 1.39{{[0-9]+}} )
//	CHECK:	( 0.68{{[0-9]+}}, 0.18{{[0-9]+}}, 0.66{{[0-9]+}}, 0.89{{[0-9]+}} )
//	CHECK:	( 0.70{{[0-9]+}}, 0.78{{[0-9]+}}, 0.29{{[0-9]+}}, 0.89{{[0-9]+}} )
//	CHECK:	( 0.67{{[0-9]+}}, 0.64{{[0-9]+}}, 0.24{{[0-9]+}}, 0.83{{[0-9]+}} )
//	CHECK:	( 0.58{{[0-9]+}}, 1.11{{[0-9]+}}, 0.65{{[0-9]+}}, 0.92{{[0-9]+}} )
//	CHECK:	( 0.47{{[0-9]+}}, 0.57{{[0-9]+}}, 0.19{{[0-9]+}}, 0.89{{[0-9]+}} )
//	CHECK:	( 0.75{{[0-9]+}}, 0.70{{[0-9]+}}, 0.48{{[0-9]+}}, 0.76{{[0-9]+}} )
//	CHECK:	( 0.91{{[0-9]+}}, 0.57{{[0-9]+}}, 0.41{{[0-9]+}}, 0.88{{[0-9]+}} )
//	CHECK:	( 0.62{{[0-9]+}}, 0.63{{[0-9]+}}, 0.41{{[0-9]+}}, 0.68{{[0-9]+}} )
//	CHECK:	( 0.67{{[0-9]+}}, 0.44{{[0-9]+}}, 0.44{{[0-9]+}}, 0.57{{[0-9]+}} )
//	CHECK:	( 0.78{{[0-9]+}}, 0.83{{[0-9]+}}, 0.51{{[0-9]+}}, 0.59{{[0-9]+}} )
//	CHECK:	( 1.16{{[0-9]+}}, 1.17{{[0-9]+}}, 0.61{{[0-9]+}}, 0.52{{[0-9]+}} )
//	CHECK:	( 0.78{{[0-9]+}}, 0.83{{[0-9]+}}, 0.47{{[0-9]+}}, 0.47{{[0-9]+}} )
//	CHECK:	( 0.81{{[0-9]+}}, 0.61{{[0-9]+}}, 0.52{{[0-9]+}}, 0.46{{[0-9]+}} )
//	CHECK:	( 1.02{{[0-9]+}}, 0.30{{[0-9]+}}, 0.88{{[0-9]+}}, 1.35{{[0-9]+}} )
//	CHECK:	( 1.08{{[0-9]+}}, 0.57{{[0-9]+}}, 0.51{{[0-9]+}}, 0.83{{[0-9]+}} )
//	CHECK:	( 0.86{{[0-9]+}}, 0.24{{[0-9]+}}, 0.73{{[0-9]+}}, 0.90{{[0-9]+}} )
//	CHECK:	( 0.85{{[0-9]+}}, 0.29{{[0-9]+}}, 0.72{{[0-9]+}}, 0.98{{[0-9]+}} )
//	CHECK:	( 0.73{{[0-9]+}}, 0.70{{[0-9]+}}, 0.37{{[0-9]+}}, 1.09{{[0-9]+}} )
//	CHECK:	( 0.81{{[0-9]+}}, 0.68{{[0-9]+}}, 0.18{{[0-9]+}}, 0.63{{[0-9]+}} )
//	CHECK:	( 0.62{{[0-9]+}}, 0.90{{[0-9]+}}, 0.38{{[0-9]+}}, 0.92{{[0-9]+}} )
//	CHECK:	( 0.76{{[0-9]+}}, 0.97{{[0-9]+}}, 0.43{{[0-9]+}}, 0.90{{[0-9]+}} )
//	CHECK:	( 1.04{{[0-9]+}}, 0.72{{[0-9]+}}, 0.60{{[0-9]+}}, 1.25{{[0-9]+}} )
//	CHECK:	( 0.70{{[0-9]+}}, 0.63{{[0-9]+}}, 0.40{{[0-9]+}}, 0.71{{[0-9]+}} )
//	CHECK:	( 0.80{{[0-9]+}}, 0.95{{[0-9]+}}, 0.55{{[0-9]+}}, 1.09{{[0-9]+}} )
//	CHECK:	( 0.80{{[0-9]+}}, 0.84{{[0-9]+}}, 0.47{{[0-9]+}}, 0.83{{[0-9]+}} )
//	CHECK:	( 1.35{{[0-9]+}}, 0.98{{[0-9]+}}, 0.72{{[0-9]+}}, 0.53{{[0-9]+}} )
//	CHECK:	( 0.69{{[0-9]+}}, 0.59{{[0-9]+}}, 0.50{{[0-9]+}}, 0.71{{[0-9]+}} )
//	CHECK:	( 0.82{{[0-9]+}}, 1.11{{[0-9]+}}, 0.81{{[0-9]+}}, 0.55{{[0-9]+}} )
//	CHECK:	( 0.87{{[0-9]+}}, 0.84{{[0-9]+}}, 0.58{{[0-9]+}}, 0.62{{[0-9]+}} )
//	CHECK:	( 0.78{{[0-9]+}}, 0.19{{[0-9]+}}, 0.61{{[0-9]+}}, 1.08{{[0-9]+}} )
//	CHECK:	( 0.92{{[0-9]+}}, 0.40{{[0-9]+}}, 0.88{{[0-9]+}}, 1.06{{[0-9]+}} )
//	CHECK:	( 1.23{{[0-9]+}}, 0.52{{[0-9]+}}, 0.68{{[0-9]+}}, 1.20{{[0-9]+}} )
//	CHECK:	( 0.95{{[0-9]+}}, 0.24{{[0-9]+}}, 0.70{{[0-9]+}}, 0.83{{[0-9]+}} )
//	CHECK:	( 0.55{{[0-9]+}}, 0.89{{[0-9]+}}, 0.22{{[0-9]+}}, 0.78{{[0-9]+}} )
//	CHECK:	( 0.68{{[0-9]+}}, 0.98{{[0-9]+}}, 0.27{{[0-9]+}}, 0.91{{[0-9]+}} )
//	CHECK:	( 0.75{{[0-9]+}}, 0.95{{[0-9]+}}, 0.40{{[0-9]+}}, 1.16{{[0-9]+}} )
//	CHECK:	( 0.70{{[0-9]+}}, 0.48{{[0-9]+}}, 0.23{{[0-9]+}}, 0.75{{[0-9]+}} )
//	CHECK:	( 0.92{{[0-9]+}}, 0.74{{[0-9]+}}, 0.79{{[0-9]+}}, 0.77{{[0-9]+}} )
//	CHECK:	( 0.75{{[0-9]+}}, 0.69{{[0-9]+}}, 0.65{{[0-9]+}}, 1.03{{[0-9]+}} )
//	CHECK:	( 1.02{{[0-9]+}}, 0.78{{[0-9]+}}, 0.58{{[0-9]+}}, 0.96{{[0-9]+}} )
//	CHECK:	( 0.89{{[0-9]+}}, 0.81{{[0-9]+}}, 1.06{{[0-9]+}}, 0.97{{[0-9]+}} )
//	CHECK:	( 0.94{{[0-9]+}}, 0.80{{[0-9]+}}, 0.53{{[0-9]+}}, 0.49{{[0-9]+}} )
//	CHECK:	( 0.97{{[0-9]+}}, 0.76{{[0-9]+}}, 0.51{{[0-9]+}}, 0.51{{[0-9]+}} )
//	CHECK:	( 1.06{{[0-9]+}}, 1.00{{[0-9]+}}, 0.55{{[0-9]+}}, 0.72{{[0-9]+}} )
//	CHECK:	( 0.81{{[0-9]+}}, 1.10{{[0-9]+}}, 0.54{{[0-9]+}}, 0.73{{[0-9]+}} )
//	CHECK:	( 0.63{{[0-9]+}}, 0.17{{[0-9]+}}, 0.49{{[0-9]+}}, 0.70{{[0-9]+}} )
//	CHECK:	( 0.73{{[0-9]+}}, 0.19{{[0-9]+}}, 0.60{{[0-9]+}}, 0.80{{[0-9]+}} )
//	CHECK:	( 1.13{{[0-9]+}}, 0.30{{[0-9]+}}, 0.70{{[0-9]+}}, 0.77{{[0-9]+}} )
//	CHECK:	( 0.86{{[0-9]+}}, 0.24{{[0-9]+}}, 0.74{{[0-9]+}}, 0.78{{[0-9]+}} )
//	CHECK:	( 0.48{{[0-9]+}}, 0.48{{[0-9]+}}, 0.29{{[0-9]+}}, 0.71{{[0-9]+}} )
//	CHECK:	( 0.61{{[0-9]+}}, 0.93{{[0-9]+}}, 0.24{{[0-9]+}}, 0.83{{[0-9]+}} )
//	CHECK:	( 0.62{{[0-9]+}}, 0.80{{[0-9]+}}, 0.50{{[0-9]+}}, 0.76{{[0-9]+}} )
//	CHECK:	( 0.61{{[0-9]+}}, 0.69{{[0-9]+}}, 0.21{{[0-9]+}}, 0.80{{[0-9]+}} )
//	CHECK:	( 0.51{{[0-9]+}}, 0.65{{[0-9]+}}, 0.36{{[0-9]+}}, 0.60{{[0-9]+}} )
//	CHECK:	( 0.72{{[0-9]+}}, 0.48{{[0-9]+}}, 0.42{{[0-9]+}}, 1.13{{[0-9]+}} )
//	CHECK:	( 0.50{{[0-9]+}}, 0.59{{[0-9]+}}, 0.36{{[0-9]+}}, 0.64{{[0-9]+}} )
//	CHECK:	( 0.84{{[0-9]+}}, 1.26{{[0-9]+}}, 0.63{{[0-9]+}}, 1.02{{[0-9]+}} )
//	CHECK:	( 0.65{{[0-9]+}}, 0.60{{[0-9]+}}, 0.44{{[0-9]+}}, 0.44{{[0-9]+}} )
//	CHECK:	( 0.83{{[0-9]+}}, 0.86{{[0-9]+}}, 0.48{{[0-9]+}}, 0.54{{[0-9]+}} )
//	CHECK:	( 1.15{{[0-9]+}}, 0.64{{[0-9]+}}, 0.68{{[0-9]+}}, 0.45{{[0-9]+}} )
//	CHECK:	( 0.72{{[0-9]+}}, 0.83{{[0-9]+}}, 0.52{{[0-9]+}}, 0.56{{[0-9]+}} )
//	CHECK:	( 0.75{{[0-9]+}}, 0.23{{[0-9]+}}, 0.48{{[0-9]+}}, 1.00{{[0-9]+}} )
//	CHECK:	( 1.13{{[0-9]+}}, 0.17{{[0-9]+}}, 0.61{{[0-9]+}}, 0.80{{[0-9]+}} )
//	CHECK:	( 1.12{{[0-9]+}}, 0.51{{[0-9]+}}, 1.17{{[0-9]+}}, 1.17{{[0-9]+}} )
//	CHECK:	( 0.95{{[0-9]+}}, 0.25{{[0-9]+}}, 0.66{{[0-9]+}}, 1.00{{[0-9]+}} )
//	CHECK:	( 0.71{{[0-9]+}}, 0.63{{[0-9]+}}, 0.18{{[0-9]+}}, 0.81{{[0-9]+}} )
//	CHECK:	( 0.69{{[0-9]+}}, 0.60{{[0-9]+}}, 0.35{{[0-9]+}}, 0.87{{[0-9]+}} )
//	CHECK:	( 0.87{{[0-9]+}}, 1.00{{[0-9]+}}, 0.49{{[0-9]+}}, 0.97{{[0-9]+}} )
//	CHECK:	( 0.71{{[0-9]+}}, 1.02{{[0-9]+}}, 0.33{{[0-9]+}}, 1.00{{[0-9]+}} )
//	CHECK:	( 0.65{{[0-9]+}}, 0.54{{[0-9]+}}, 0.38{{[0-9]+}}, 0.60{{[0-9]+}} )
//	CHECK:	( 0.60{{[0-9]+}}, 0.56{{[0-9]+}}, 0.34{{[0-9]+}}, 1.10{{[0-9]+}} )
//	CHECK:	( 0.98{{[0-9]+}}, 0.93{{[0-9]+}}, 0.59{{[0-9]+}}, 1.51{{[0-9]+}} )
//	CHECK:	( 0.87{{[0-9]+}}, 0.99{{[0-9]+}}, 0.53{{[0-9]+}}, 0.83{{[0-9]+}} )
//	CHECK:	( 1.02{{[0-9]+}}, 0.72{{[0-9]+}}, 0.41{{[0-9]+}}, 0.39{{[0-9]+}} )
//	CHECK:	( 1.08{{[0-9]+}}, 0.62{{[0-9]+}}, 0.61{{[0-9]+}}, 0.50{{[0-9]+}} )
//	CHECK:	( 1.03{{[0-9]+}}, 1.09{{[0-9]+}}, 0.58{{[0-9]+}}, 0.64{{[0-9]+}} )
//	CHECK:	( 1.24{{[0-9]+}}, 0.92{{[0-9]+}}, 0.69{{[0-9]+}}, 0.78{{[0-9]+}} )
//	CHECK:	( 1.53{{[0-9]+}}, 0.28{{[0-9]+}}, 0.87{{[0-9]+}}, 1.45{{[0-9]+}} )
//	CHECK:	( 0.85{{[0-9]+}}, 0.16{{[0-9]+}}, 0.59{{[0-9]+}}, 0.94{{[0-9]+}} )
//	CHECK:	( 0.91{{[0-9]+}}, 0.31{{[0-9]+}}, 0.75{{[0-9]+}}, 1.15{{[0-9]+}} )
//	CHECK:	( 1.01{{[0-9]+}}, 0.49{{[0-9]+}}, 0.64{{[0-9]+}}, 1.05{{[0-9]+}} )
//	CHECK:	( 1.21{{[0-9]+}}, 0.74{{[0-9]+}}, 0.49{{[0-9]+}}, 1.58{{[0-9]+}} )
//	CHECK:	( 0.70{{[0-9]+}}, 0.63{{[0-9]+}}, 0.33{{[0-9]+}}, 0.95{{[0-9]+}} )
//	CHECK:	( 0.84{{[0-9]+}}, 0.79{{[0-9]+}}, 0.32{{[0-9]+}}, 1.09{{[0-9]+}} )
//	CHECK:	( 0.80{{[0-9]+}}, 0.78{{[0-9]+}}, 0.75{{[0-9]+}}, 1.00{{[0-9]+}} )
//	CHECK:	( 1.12{{[0-9]+}}, 0.81{{[0-9]+}}, 0.86{{[0-9]+}}, 1.14{{[0-9]+}} )
//	CHECK:	( 0.61{{[0-9]+}}, 0.53{{[0-9]+}}, 0.36{{[0-9]+}}, 0.53{{[0-9]+}} )
//	CHECK:	( 0.92{{[0-9]+}}, 0.85{{[0-9]+}}, 0.59{{[0-9]+}}, 0.91{{[0-9]+}} )
//	CHECK:	( 1.10{{[0-9]+}}, 1.10{{[0-9]+}}, 0.52{{[0-9]+}}, 0.99{{[0-9]+}} )
//	CHECK:	( 1.08{{[0-9]+}}, 0.99{{[0-9]+}}, 1.03{{[0-9]+}}, 0.79{{[0-9]+}} )
//	CHECK:	( 0.72{{[0-9]+}}, 0.78{{[0-9]+}}, 0.43{{[0-9]+}}, 0.45{{[0-9]+}} )
//	CHECK:	( 0.96{{[0-9]+}}, 0.91{{[0-9]+}}, 0.87{{[0-9]+}}, 0.59{{[0-9]+}} )
//	CHECK:	( 1.02{{[0-9]+}}, 0.95{{[0-9]+}}, 0.70{{[0-9]+}}, 0.64{{[0-9]+}} )

