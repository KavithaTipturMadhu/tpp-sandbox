#include "TPP/BuilderUtils.h"
#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/Dialect/Tpp/TppUtils.h"
#include "TPP/Passes.h"
#include "TPP/TransformUtils.h"
#include "TPP/Transforms.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"
#include <iostream>
namespace {

typedef struct MatchResult {
  bool haveMatch;
  int index;
} MatchResult;

template <typename PackOp>
SmallVector<MatchResult> innerDimMatchesIndex(PackOp packOp) {
  SmallVector<MatchResult> result;
  int innerTiles = 0;
  for (size_t i = 0; i < packOp.getSource().getType().getShape().size(); i++) {
    if (packOp.getInnerDimsPos().size() > 0) {
      bool haveMatch = false;
      for (size_t j = 0; j < packOp.getInnerDimsPos().size(); j++) {
        if (i == (size_t)packOp.getInnerDimsPos()[j]) {
          MatchResult tempResult;
          tempResult.haveMatch = true;
          haveMatch = true;
          tempResult.index = innerTiles++;
          result.push_back(tempResult);
          break;
        }
      }
      if (!haveMatch) {
        MatchResult tempResult;
        tempResult.haveMatch = false;
        tempResult.index = -1;
        result.push_back(tempResult);
      }
    }
  }
  return result;
}
struct ConvertTppPackOp : public OpRewritePattern<tensor::PackOp> {
  using OpRewritePattern<tensor::PackOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    if (packOp.getStaticInnerTiles().size() > 0) {
      int numLoops = packOp.getSource().getType().getShape().size();

      SmallVector<MatchResult> haveMatched = innerDimMatchesIndex(packOp);
      for (int i = 0; i < numLoops; i++) {
        if (haveMatched[i].haveMatch) {
          if (packOp.getSource().getType().getShape()[i] <
              packOp.getStaticInnerTiles()[haveMatched[i].index]) {
            return failure();
          }
        }
      }

      auto zero = getConstIndex(rewriter, 0);
      auto one = getConstIndex(rewriter, 1);

      SmallVector<Value> lbs;
      SmallVector<Value> ubs;
      SmallVector<Value> steps;

      for (int i = 0; i < numLoops; i++) {
        lbs.push_back(zero);
        steps.push_back(one);
      }

      for (int i = 0; i < numLoops; i++) {

        if (haveMatched[i].haveMatch) {
          ubs.push_back(getConstIndex(
              rewriter,
              packOp.getSource().getType().getShape()[i] /
                  packOp.getStaticInnerTiles()[haveMatched[i].index]));

        } else {
          ubs.push_back(getConstIndex(
              rewriter, packOp.getSource().getType().getShape()[i]));
        }
      }
      Value pos;

      SmallVector<Value> reduc = {
          packOp.getDest(),
      };

      auto bodyBuilder = [&](OpBuilder &builder, Location, Value iv,
                             MutableArrayRef<Value> reduc) {};

      auto loopNest = mlir::scf::buildLoopNest(
          rewriter, packOp.getLoc(), lbs, ubs, steps, reduc,
          [&reduc, &pos, bodyBuilder, &packOp,
           &numLoops](OpBuilder &rewriter, Location loc, ValueRange localIvs,
                      ValueRange iterArgs) -> scf::ValueVector {
            reduc.assign(iterArgs.begin(), iterArgs.end());

            SmallVector<OpFoldResult> offsets;
            SmallVector<MatchResult> haveMatched = innerDimMatchesIndex(packOp);
            for (int i = 0; i < numLoops; i++) {

              if (haveMatched[i].haveMatch) {
                Value muliOp = rewriter.create<arith::MulIOp>(
                    loc, localIvs[i],
                    getConstIndex(
                        rewriter,
                        packOp.getStaticInnerTiles()[haveMatched[i].index]));
                offsets.push_back(muliOp);
              } else {
                offsets.push_back(localIvs[i]);
              }
            }
            SmallVector<OpFoldResult> strides;
            for (int i = 0; i < numLoops; i++) {
              strides.push_back(rewriter.getIndexAttr(1));
            }

            SmallVector<OpFoldResult> sizes;

            for (int i = 0; i < numLoops; i++) {
              if (haveMatched[i].haveMatch) {
                sizes.push_back(rewriter.getIndexAttr(
                    packOp.getStaticInnerTiles()[haveMatched[i].index]));
              } else {
                sizes.push_back(rewriter.getIndexAttr(1));
              }
            }

            auto tensorExtractType =
                tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
                    packOp.getStaticInnerTiles().size(),
                    packOp.getSource().getType().cast<RankedTensorType>(),
                    offsets, sizes, strides);

            auto tensorExtract = rewriter.create<tensor::ExtractSliceOp>(
                loc, tensorExtractType.cast<RankedTensorType>(),
                packOp.getSource(), offsets, sizes, strides);

            SmallVector<OpFoldResult> insertSliceOffsets;
            for (int i = 0; i < numLoops; i++) {
              int indirection = i;
              if (packOp.getOuterDimsPerm().size() > 0) {
                indirection = packOp.getOuterDimsPerm()[i];
              }
              insertSliceOffsets.push_back(localIvs[indirection]);
            }
            for (size_t i = numLoops; i < packOp.getDestRank(); i++) {
              insertSliceOffsets.push_back(rewriter.getIndexAttr(0));
            }
            SmallVector<OpFoldResult> insertSliceSizes;
            for (int i = 0; i < numLoops; i++) {
              insertSliceSizes.push_back(rewriter.getIndexAttr(1));
            }
            for (size_t i = numLoops; i < packOp.getDestRank(); i++) {
              insertSliceSizes.push_back(rewriter.getIndexAttr(
                  packOp.getStaticInnerTiles()[i - numLoops]));
            }

            SmallVector<OpFoldResult> insertSliceStrides(
                packOp.getDestRank(), rewriter.getIndexAttr(1));
            auto insertSliceOp = rewriter.create<tensor::InsertSliceOp>(
                loc, tensorExtract.getResult(), iterArgs[0], insertSliceOffsets,
                insertSliceSizes, insertSliceStrides);
            return {insertSliceOp};
          });
      rewriter.replaceAllUsesWith(packOp.getResult(),
                                  loopNest.loops[0].getResults()[0]);
      return success();
    }
    return failure();
  }
};

struct ConvertTppUnpackOp : public OpRewritePattern<tensor::UnPackOp> {
  using OpRewritePattern<tensor::UnPackOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::UnPackOp unpackOp,
                                PatternRewriter &rewriter) const override {
    if (unpackOp.getStaticInnerTiles().size() > 0) {
      int numLoops = unpackOp.getDest().getType().getShape().size();

      SmallVector<MatchResult> haveMatched = innerDimMatchesIndex(unpackOp);
      for (int i = 0; i < numLoops; i++) {
        if (haveMatched[i].haveMatch) {
          if (unpackOp.getDest().getType().getShape()[i] <
              unpackOp.getStaticInnerTiles()[haveMatched[i].index]) {
            return failure();
          }
        }
      }

      auto zero = getConstIndex(rewriter, 0);
      auto one = getConstIndex(rewriter, 1);

      SmallVector<Value> lbs;
      SmallVector<Value> ubs;
      SmallVector<Value> steps;

      for (int i = 0; i < numLoops; i++) {
        lbs.push_back(zero);
        steps.push_back(one);
      }

      for (int i = 0; i < numLoops; i++) {
        if (haveMatched[i].haveMatch) {
          ubs.push_back(getConstIndex(
              rewriter,
              unpackOp.getDest().getType().getShape()[i] /
                  unpackOp.getStaticInnerTiles()[haveMatched[i].index]));

        } else {
          ubs.push_back(getConstIndex(
              rewriter, unpackOp.getDest().getType().getShape()[i]));
        }
      }
      Value pos;

      SmallVector<Value> reduc = {
          unpackOp.getDest(),
      };

      auto bodyBuilder = [&](OpBuilder &builder, Location, Value iv,
                             MutableArrayRef<Value> reduc) {};

      auto loopNest = mlir::scf::buildLoopNest(
          rewriter, unpackOp.getLoc(), lbs, ubs, steps, reduc,
          [&reduc, &pos, bodyBuilder, &unpackOp,
           &numLoops](OpBuilder &rewriter, Location loc, ValueRange localIvs,
                      ValueRange iterArgs) -> scf::ValueVector {
            reduc.assign(iterArgs.begin(), iterArgs.end());

            SmallVector<OpFoldResult> offsets;
            SmallVector<MatchResult> haveMatched = innerDimMatchesIndex(unpackOp);
            for (int i = 0; i < numLoops; i++) {

              if (haveMatched[i].haveMatch) {
                Value muliOp = rewriter.create<arith::MulIOp>(
                    loc, localIvs[i],
                    getConstIndex(
                        rewriter,
                        unpackOp.getStaticInnerTiles()[haveMatched[i].index]));
                offsets.push_back(muliOp);
              } else {
                offsets.push_back(localIvs[i]);
              }
            }
            SmallVector<OpFoldResult> strides;
            for (int i = 0; i < numLoops; i++) {
              strides.push_back(rewriter.getIndexAttr(1));
            }

            SmallVector<OpFoldResult> sizes;

            for (int i = 0; i < numLoops; i++) {
              if (haveMatched[i].haveMatch) {
                sizes.push_back(rewriter.getIndexAttr(
                    unpackOp.getStaticInnerTiles()[haveMatched[i].index]));
              } else {
                sizes.push_back(rewriter.getIndexAttr(1));
              }
            }
            SmallVector<OpFoldResult> extractSliceOffsets;
            for (int i = 0; i < numLoops; i++) {
              int indirection = i;
              if (unpackOp.getOuterDimsPerm().size() > 0) {
                indirection = unpackOp.getOuterDimsPerm()[i];
              }
              extractSliceOffsets.push_back(localIvs[indirection]);
            }
            for (size_t i = numLoops; i < unpackOp.getSourceRank(); i++) {
              extractSliceOffsets.push_back(rewriter.getIndexAttr(0));
            }
            SmallVector<OpFoldResult> extractSliceSizes;
            for (int i = 0; i < numLoops; i++) {
              extractSliceSizes.push_back(rewriter.getIndexAttr(1));
            }
            for (size_t i = numLoops; i < unpackOp.getSourceRank(); i++) {
              extractSliceSizes.push_back(rewriter.getIndexAttr(
                  unpackOp.getStaticInnerTiles()[i - numLoops]));
            }
	    SmallVector<OpFoldResult> extractSliceStrides(
                unpackOp.getSourceRank(), rewriter.getIndexAttr(1));
            auto tensorExtractType =
                tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
                    unpackOp.getStaticInnerTiles().size(),
                    unpackOp.getSource().getType().cast<RankedTensorType>(),
                    extractSliceOffsets, extractSliceSizes, extractSliceStrides);

            auto tensorExtract = rewriter.create<tensor::ExtractSliceOp>(
                loc, tensorExtractType.cast<RankedTensorType>(),
                unpackOp.getSource(), extractSliceOffsets, extractSliceSizes, extractSliceStrides);

            auto insertSliceOp = rewriter.create<tensor::InsertSliceOp>(
                loc, tensorExtract.getResult(), iterArgs[0], offsets,
                sizes, strides);
            return {insertSliceOp};
          });
      rewriter.replaceAllUsesWith(unpackOp.getResult(),
                                  loopNest.loops[0].getResults()[0]);
      return success();
    }
    return failure();
  }
};

void populateTppPackUnpackPatterns(RewritePatternSet &patterns) {
  // clang-format off
     patterns.add<ConvertTppPackOp>(patterns.getContext());
     patterns.add<ConvertTppUnpackOp>(patterns.getContext());
  // clang-format on
}

struct ConvertTppPackUnpack
    : public ConvertTppToIdentityBase<ConvertTppPackUnpack> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateTppPackUnpackPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
    registry.insert<tpp::TppDialect>();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createConvertTppPackUnpack() {
  return std::make_unique<ConvertTppPackUnpack>();
}
