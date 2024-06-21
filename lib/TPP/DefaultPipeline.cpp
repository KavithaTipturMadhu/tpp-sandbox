//===- DefaultPipeline.cpp ---------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/PassBundles.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"

#include "TPP/Dialect/Check/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Check/CheckDialect.h"
#include "TPP/Dialect/Perf/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Perf/PerfDialect.h"
#include "TPP/Dialect/Perf/PerfOps.h"
#include "TPP/Dialect/Xsmm/XsmmDialect.h"
#include "TPP/PassUtils.h"
#include "mlir/Transforms/Passes.h"

#include <string>

using namespace mlir;
using namespace mlir::tpp;

// Print MLIR before lowering
llvm::cl::opt<std::string>
    printMLIR("print-mlir",
              llvm::cl::desc("Print MLIR to stdout (early, mid, late, llvm)"),
              llvm::cl::init(""));

// Lower Linalg directly to loops without TPP (for validation purposes)
llvm::cl::opt<bool> linalgToLoops("linalg-to-loops",
                                  llvm::cl::desc("Lower linalg to loops"),
                                  llvm::cl::init(false));

// Control parallelism.
llvm::cl::opt<bool>
    defParallel("def-parallel",
                llvm::cl::desc("Default pipeline - enable parallel execution"),
                llvm::cl::init(false));

// Control grid parallelism sizes.
llvm::cl::list<unsigned>
    parallelTaskGrid("parallel-task-grid",
                     llvm::cl::desc("Grid-sizes for parallel tasks"),
                     llvm::cl::list_init<unsigned>(SmallVector<unsigned>{2, 8}),
                     llvm::cl::CommaSeparated);

llvm::cl::opt<unsigned> tileShapeM(
    "M-tile-shape",
    llvm::cl::desc(
        "Shape to tile the first tiled input operand tensor of brgemm op by"),
    llvm::cl::init(0));

llvm::cl::opt<unsigned> tileShapeN(
    "N-tile-shape",
    llvm::cl::desc(
        "Shape to tile the second tiled input operand tensor of brgemm op by"),
    llvm::cl::init(0));

llvm::cl::list<unsigned> shuffleOrder(
    "loop-shuffle-order",
    llvm::cl::desc("Permutation shuffle order (integer index list) to shuffle "
                   "the scf forall loop surrounding brgemm op"),
    llvm::cl::CommaSeparated);

llvm::cl::opt<unsigned> outerParallelLoops(
    "num-outer-parallel",
    llvm::cl::desc("Number of outer loops to be parallelized"),
    llvm::cl::value_desc("int"), llvm::cl::init(0));

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_DEFAULTPIPELINE
#include "TPP/PassBundles.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Enum to control IR printing.
enum class PrintStage {
  None,
  Early, // After main generation, before optimization
  Mid,   // After initial TPP-related optimizations
  Late,  // After optimizaiton, before LLVM dialect
  LLVM,  // Final MLIR, in LLVM dialect
};

// Parses MLIR print stage
PrintStage parsePrintStage(StringRef stage) {
  return StringSwitch<PrintStage>(stage)
      .CaseLower("early", PrintStage::Early)
      .CaseLower("mid", PrintStage::Mid)
      .CaseLower("late", PrintStage::Late)
      .CaseLower("llvm", PrintStage::LLVM)
      .Default(PrintStage::None);
}

// The default lowering pipeline.
struct DefaultPipeline : public tpp::impl::DefaultPipelineBase<DefaultPipeline>,
                         PassBundle<ModuleOp> {
  using DefaultPipelineBase::DefaultPipelineBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    // Add all custom TPP dialects.
    registry.insert<xsmm::XsmmDialect>();
    registry.insert<check::CheckDialect>();
    registry.insert<perf::PerfDialect>();
    check::registerBufferizableOpInterfaceExternalModels(registry);
    perf::registerBufferizableOpInterfaceExternalModels(registry);

    // Add all core MLIR dialects as the default pipeline may contain any
    // combination of other passes.
    registerAllDialects(registry);
  }

  void runOnOperation() override {
    auto module = getOperation();

    // Initialize the pipeline if needed.
    // Otherwise, just run the cached one.
    if (pm.empty())
      constructPipeline();

    if (failed(runPipeline(pm, module)))
      return signalPassFailure();
  }

private:
  void constructPipeline() override {
    auto print = parsePrintStage(printMLIR);

    // Print IR of unoptimized kernel and main
    if (print == PrintStage::Early)
      pm.addPass(createPrintIRPass());

    if (!gpuBackend.empty()) {
      // Apply the custom GPU lowering pipeline
      pm.addPass(createGpuPipeline(GpuPipelineOptions{gpuBackend}));
    } else {
      // Apply the default preprocessing pass
      DefaultTppPassesOptions tppDefaultOptions;
      tppDefaultOptions.tileShapeM = tileShapeM;
      tppDefaultOptions.tileShapeN = tileShapeN;
      if (!shuffleOrder.empty())
        tppDefaultOptions.shuffleOrder = shuffleOrder;
      tppDefaultOptions.linalgToLoops = linalgToLoops;
      tppDefaultOptions.parallelTaskGrid = parallelTaskGrid;
      tppDefaultOptions.outerParallelLoops = outerParallelLoops;
      pm.addPass(createDefaultTppPasses(tppDefaultOptions));
    }

    if (print == PrintStage::Mid)
      pm.addPass(createPrintIRPass());

    // Bail out early for Intel GPU.
    // The rest of the lowering is performed by IMEX.
    if (gpuBackend == "intel") {
      pm.addPass(createPrintIRPass());
      return;
    }

    // Partial Lowering
    pm.addPass(memref::createExpandStridedMetadataPass());
    pm.addPass(createConvertTensorToLinalgPass());
    pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
    if (defParallel)
      pm.addPass(createConvertSCFToOpenMPPass());
    pm.addPass(createConvertVectorToSCFPass());
    pm.addPass(arith::createArithExpandOpsPass());
    pm.addPass(createLowerAffinePass());

    // Print IR of optimized kernel and main
    if (print == PrintStage::Late)
      pm.addPass(createPrintIRPass());

    // Lower to LLVM
    pm.addPass(createConvertVectorToLLVMPass());
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(createConvertSCFToCFPass());
    if (defParallel)
      pm.addPass(createConvertOpenMPToLLVMPass());
    pm.addPass(createConvertMathToLLVMPass());

    pm.addNestedPass<func::FuncOp>(createGpuAsyncRegionPass());
    pm.addPass(createGpuToLLVMConversionPass());
    GpuModuleToBinaryPassOptions gpuModuleToBinaryPassOptions;
    gpuModuleToBinaryPassOptions.compilationTarget = "fatbin";
    pm.addPass(createGpuModuleToBinaryPass(gpuModuleToBinaryPassOptions));
    pm.addPass(createAsyncToAsyncRuntimePass());
    pm.addPass(createAsyncRuntimeRefCountingPass());
    pm.addPass(createConvertAsyncToLLVMPass());

    pm.addPass(createConvertFuncToLLVMPass());

    pm.addNestedPass<func::FuncOp>(createArithToLLVMConversionPass());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(createCSEPass());
    pm.addPass(createReconcileUnrealizedCastsPass());

    pm.addPass(createConvertVulkanLaunchFuncToVulkanCallsPass());

    // Anything useful has been lowered by now.
    // Cleanup IR by removing any dead symbols.
    // This step aims to avoid errors caused by frontend leftovers.
    // See issue: #704
    pm.addPass(createSymbolDCEPass());

    // Print IR of kernel and main in LLVM dialect
    if (print == PrintStage::LLVM)
      pm.addPass(createPrintIRPass());
  }
};

} // namespace
