//===- Transforms.h ----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class RewriterBase;
class Value;

namespace linalg {
class GenericOp;
class LinalgOp;
} // namespace linalg

namespace tpp {

// Attempt to map the current linalgOp to a BRGEMM.
// On success the returned values are the materialzed loops with BRGEMM inside.
FailureOr<SmallVector<Value>> MapToBRGEMMOp(RewriterBase &rewriter,
                                            linalg::LinalgOp linalgOp);

// Attempt to block a Conv2DNchwFchwOp.
FailureOr<linalg::GenericOp>
BlockConv2DNchwFchwOp(RewriterBase &rewriter, linalg::LinalgOp linalgOp,
                      ArrayRef<int64_t> blockingFactors);

} // namespace tpp
} // namespace mlir
