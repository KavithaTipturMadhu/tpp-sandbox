//===- LoopInsertion.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements parallel loop insertion for tiling.
//
//===----------------------------------------------------------------------===//
#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "loop-insertion"

namespace mlir {
namespace tpp {
#define GEN_PASS_DECL_LOOPINSERTIONPASS
#define GEN_PASS_DEF_LOOPINSERTIONPASS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;

namespace mlir {
namespace tpp {

static SmallVector<ReassociationIndices>
getReassociationIndices(ArrayRef<int64_t> origtensorShape,
                        SmallVector<SmallVector<unsigned>> tileShapes) {
  SmallVector<ReassociationIndices> indices;

  size_t index = 0;
  for (size_t i = 0; i < tileShapes.size(); i++) {
    ReassociationIndices reassociationIndex;
    for (size_t j = 0; j < tileShapes[i].size(); j++)
      reassociationIndex.push_back(index++);
    indices.push_back(reassociationIndex);
  }
  for (size_t i = tileShapes.size(); i < origtensorShape.size(); i++) {
    ReassociationIndices reassociationIndex;
    reassociationIndex.push_back(index++);
    indices.push_back(reassociationIndex);
  }

  return indices;
}

static void insertSubview(ArrayRef<int64_t> tensorShape, Type type,
                          Type resultType,
                          SmallVector<ReassociationIndices> reassociation,
                          Value operand, ForallOp op, IRRewriter &rewriter,
                          Operation *originalSubviewOp,
                          int inductionVarIndexBase) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  auto expandShape = rewriter.create<memref::ExpandShapeOp>(
      op.getLoc(),
      MemRefType::get({tensorShape},
                      dyn_cast<MemRefType>(type).getElementType()),
      operand, reassociation);
  expandShape.setStaticOutputShape(tensorShape);
  rewriter.setInsertionPointToStart(op.getBody());
  SmallVector<OpFoldResult> strides(tensorShape.size(),
                                    rewriter.getIndexAttr(1)),
      sizes, offsets;

  size_t tileSize =
      tensorShape.size() - dyn_cast<ShapedType>(resultType).getShape().size();

  SmallVector<int64_t> tileSizes;
  for (size_t i = 0; i < tileSize; i++) {
    int inductionVarIndex = inductionVarIndexBase * tileSize + i;
    offsets.push_back(op.getInductionVars()[inductionVarIndex]);
    sizes.push_back(rewriter.getIndexAttr(1));
  }

  for (size_t i = tileSize; i < tensorShape.size(); i++) {
    sizes.push_back(rewriter.getIndexAttr(tensorShape[i]));
    tileSizes.push_back(tensorShape[i]);
    offsets.push_back(rewriter.getIndexAttr(0));
  }

  auto subviewType =
      MemRefType::get({tileSizes}, dyn_cast<MemRefType>(type).getElementType());
  auto [originalStride, originalOffset] =
      getStridesAndOffset(dyn_cast<MemRefType>(subviewType));
  subviewType = MemRefType::get(
      {tileSizes}, dyn_cast<MemRefType>(subviewType).getElementType(),
      StridedLayoutAttr::get(rewriter.getContext(), ShapedType::kDynamic,
                             originalStride));
  auto subview = rewriter.create<memref::SubViewOp>(
      op.getLoc(), dyn_cast<MemRefType>(subviewType), expandShape.getResult(),
      offsets, sizes, strides);
  originalSubviewOp->replaceAllUsesWith(subview);
}

static LogicalResult insertParallelLoop(ForallOp op, unsigned mTileShape,
                                        unsigned nTileShape) {
  xsmm::BrgemmOp brgemmOp = NULL;
  OpBuilder b(op);
  IRRewriter rewriter(b.getContext());
  for (auto oper = op.getBody()->getOperations().begin();
       oper != op.getBody()->getOperations().end(); oper++)
    if (dyn_cast<xsmm::BrgemmOp>(oper)) {
      brgemmOp = dyn_cast<xsmm::BrgemmOp>(oper);
      break;
    }
  if (brgemmOp == NULL)
    return rewriter.notifyMatchFailure(op, "require brgemm op");

  if (mTileShape == 0 || nTileShape == 0)
    return rewriter.notifyMatchFailure(op, "require tile shape to not be zero");

  ArrayRef<int64_t> mShape;
  auto mDefiningOp = brgemmOp.getOperand(1).getDefiningOp();
  if (isa<memref::SubViewOp>(mDefiningOp))
    mShape =
        dyn_cast<ShapedType>(mDefiningOp->getOperand(0).getType()).getShape();
  else
    mShape =
        dyn_cast<ShapedType>(mDefiningOp->getResult(0).getType()).getShape();

  ArrayRef<int64_t> nShape;
  auto nDefiningOp = brgemmOp.getOperand(2).getDefiningOp();
  if (isa<memref::SubViewOp>(nDefiningOp))
    nShape =
        dyn_cast<ShapedType>(nDefiningOp->getOperand(0).getType()).getShape();
  else
    nShape =
        dyn_cast<ShapedType>(nDefiningOp->getResult(0).getType()).getShape();

  ArrayRef<int64_t> kShape;
  auto kDefiningOp = brgemmOp.getOperand(3).getDefiningOp();
  if (isa<memref::SubViewOp>(kDefiningOp))
    kShape =
        dyn_cast<ShapedType>(kDefiningOp->getOperand(0).getType()).getShape();
  else
    kShape =
        dyn_cast<ShapedType>(kDefiningOp->getResult(0).getType()).getShape();

  if (mShape.size() != kShape.size())
    return rewriter.notifyMatchFailure(
        op, "require m tensor shape and k tensor shape to match");

  SmallVector<unsigned> tileShapeM;
  tileShapeM.push_back(mTileShape);
  tileShapeM.push_back(mShape[0] / mTileShape);

  SmallVector<unsigned> tileShapeN;
  tileShapeN.push_back(nTileShape);
  tileShapeN.push_back(nShape[0] / nTileShape);

  // Validate the input tile sizes against the operand shapes
  long multipleM = 1;
  for (size_t i = 0; i < tileShapeM.size(); i++)
    multipleM = multipleM * tileShapeM[i];

  if (mShape[0] != multipleM)
    return rewriter.notifyMatchFailure(
        op, "require m tile shape to match tensor shape");

  long multipleN = 1;
  for (size_t i = 0; i < tileShapeN.size(); i++)
    multipleN = multipleN * tileShapeN[i];

  if (nShape[0] != multipleN)
    return rewriter.notifyMatchFailure(
        op, "require n tile shape to match tensor shape");

  if ((multipleM * multipleN) != (kShape[0] * kShape[1]))
    return rewriter.notifyMatchFailure(
        op, "require k tile shape to match tensor shape");

  int boundSize = tileShapeM.size() + tileShapeN.size();

  // Set the new bounds of for loop
  SmallVector<int64_t> lbs(boundSize, 0), steps(boundSize, 1);

  SmallVector<int64_t> ubs(tileShapeM.begin(), tileShapeM.end());
  ubs.append(tileShapeN.begin(), tileShapeN.end());

  op.setStaticLowerBound(lbs);
  op.setStaticUpperBound(ubs);
  op.setStaticStep(steps);

  // Add new induction var args to the for loop
  int numArgs = op.getBody()->getArguments().size();

  for (int i = 0; i < boundSize - numArgs; i++)
    op.getBody()->addArgument(b.getIndexType(), op.getLoc());

  SmallVector<int64_t> tileOffsets{
      0, static_cast<int64_t>(tileShapeM.size()),
      static_cast<int64_t>(tileShapeN.size() + tileShapeM.size())};
  rewriter.setInsertionPointToStart(op.getBody());

  SmallVector<ArrayRef<int64_t>> originalShapes{mShape, nShape, kShape};
  SmallVector<SmallVector<SmallVector<unsigned>>> tilingVectors{
      {tileShapeM}, {tileShapeN}, {tileShapeM, tileShapeN}};

  // Replace old args with newly computed args
  for (auto oper = op.getBody()->getOperations().begin();
       oper != op.getBody()->getOperations().end(); oper++) {
    int operandIndex = 0;
    for (auto arg : oper->getOperands()) {
      int oldArgIndex = -1;
      for (int i = 0; i < numArgs; i++) {
        if (arg == op.getInductionVar(i)) {
          oldArgIndex = i;
          break;
        }
      }
      if (oldArgIndex != -1) {
        Value add = op.getBody()->getArgument(tileOffsets[oldArgIndex]);
        Value mul;
        for (int j = tileOffsets[oldArgIndex] + 1;
             j < tileOffsets[oldArgIndex + 1]; j++) {
          unsigned cumulativeUpperBound = 1;
          for (int k = j; k < tileOffsets[oldArgIndex + 1]; k++)
            cumulativeUpperBound *= op.getStaticUpperBound()[j];
          Value upperBound = rewriter.create<arith::ConstantIndexOp>(
              op.getLoc(), cumulativeUpperBound);
          mul = rewriter.create<arith::MulIOp>(op.getLoc(), b.getIndexType(),
                                               add, upperBound);
          add = rewriter.create<arith::AddIOp>(
              op.getLoc(), b.getIndexType(), mul, op.getBody()->getArgument(j));
        }
        oper->setOperand(operandIndex, add);
      }
      operandIndex++;
    }
  }

  for (int i = 1; i <= 3; i++) {
    auto definingOp = brgemmOp.getOperand(i).getDefiningOp();
    if (isa<memref::SubViewOp>(definingOp)) {
      Value operand = definingOp->getOperand(0);

      auto operandType = operand.getType();
      auto resultType = dyn_cast<MemRefType>(brgemmOp.getOperand(i).getType());
      auto reassociationIndex =
          getReassociationIndices(originalShapes[i - 1], tilingVectors[i - 1]);

      SmallVector<int64_t> shape;
      for (size_t j = 0; j < tilingVectors[i - 1].size(); j++)
        shape.append(tilingVectors[i - 1][j].begin(),
                     tilingVectors[i - 1][j].end());
      shape.append(
          std::next(originalShapes[i - 1].begin(), tilingVectors[i - 1].size()),
          originalShapes[i - 1].end());
      int inductionVarBase = i - 1;
      if (definingOp == brgemmOp.getOperand(3).getDefiningOp())
        inductionVarBase = 0;
      insertSubview(shape, operandType, resultType, reassociationIndex, operand,
                    op, rewriter, definingOp, inductionVarBase);
    }
  }

  return success();
}

bool getInnermostForLoops(Operation *rootOp,
                          SmallVectorImpl<scf::ForallOp> &result) {
  assert(rootOp != nullptr && "Root operation must not be a nullptr.");
  bool rootEnclosesForAllloops = false;
  for (Region &region : rootOp->getRegions()) {
    for (Block &block : region.getBlocks()) {
      for (Operation &op : block) {
        bool enclosesPloops = getInnermostForLoops(&op, result);
        rootEnclosesForAllloops |= enclosesPloops;
        if (auto ploop = dyn_cast<scf::ForallOp>(op)) {
          rootEnclosesForAllloops = true;

          // Collect forall loop if it is an innermost one.
          if (!enclosesPloops)
            result.push_back(ploop);
        }
      }
    }
  }
  return rootEnclosesForAllloops;
}

struct LoopInsertionPass
    : public tpp::impl::LoopInsertionPassBase<LoopInsertionPass> {

  using LoopInsertionPassBase::LoopInsertionPassBase;

  void runOnOperation() override {
    auto *parentOp = getOperation();
    SmallVector<ForallOp> innermostForAllloops;
    getInnermostForLoops(parentOp, innermostForAllloops);
    for (ForallOp loop : innermostForAllloops)
      if (failed(insertParallelLoop(loop, tileShapeM, tileShapeN)))
        LLVM_DEBUG(llvm::dbgs() << "Failed to tile the loop\n");
  }
};
} // namespace tpp
} // namespace mlir
