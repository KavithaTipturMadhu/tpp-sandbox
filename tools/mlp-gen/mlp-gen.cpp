//===- mlp-gen MLP Generator ----------------------------------------------===//
//
// Main entry-point to the MLP generator. Creates an MLP model with input,
// output and multiple hidden layers, different activation functions, etc.
// Handles multiple tensor sizes, conversion, broadcast, etc.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/ValueRange.h"

#include "MLPGen.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

using namespace mlir;

// Input layer
llvm::cl::opt<unsigned> miniBatch("mini-batch",
                                   llvm::cl::desc("Mini batch size"),
                                   llvm::cl::value_desc("256"),
                                   llvm::cl::init(256));

// Hidden layers
llvm::cl::opt<std::string> layers(
    "layers",
    llvm::cl::desc("Comma-separated values of size of each layer (at least 2)"),
    llvm::cl::value_desc("128,256,512"), llvm::cl::init("128,256,512"));

// Tile sizes (N, C, K)
llvm::cl::opt<std::string> tiles(
    "tiles",
    llvm::cl::desc("Comma-separated values of size of each tile (N,K,C)"),
    llvm::cl::value_desc("32,32,32"), llvm::cl::init(""));

// Float width
llvm::cl::opt<unsigned> floatWidth("float-width",
                                   llvm::cl::desc("Bitsize of float type"),
                                   llvm::cl::value_desc("32|16"),
                                   llvm::cl::init(32));

// Random seed
llvm::cl::opt<int> seed("seed", llvm::cl::desc("Random seed"),
                             llvm::cl::value_desc("int"), llvm::cl::init(0));

// Output filename
llvm::cl::opt<std::string> filename("o", llvm::cl::desc("Output filename"),
                                    llvm::cl::value_desc("stdout"),
                                    llvm::cl::init("-"));

// Enable softmax at the last layer
llvm::cl::opt<bool> enableSoftmax("softmax", llvm::cl::desc("Enable softmax"),
                                    llvm::cl::value_desc("bool"),
                                    llvm::cl::init(false));

// Initialize accumulation matrix with bias
llvm::cl::opt<bool> biasAcc("bias-acc", llvm::cl::desc("Accumulate on bias"),
                            llvm::cl::value_desc("bool"),
                            llvm::cl::init(false));

// Enable optimal packing (including VNNI for BF16)
llvm::cl::opt<bool> pack("pack", llvm::cl::desc("Optimal packing (+VNNI)"),
                            llvm::cl::value_desc("bool"),
                            llvm::cl::init(false));

int main(int argc, char **argv) {
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();

  llvm::cl::ParseCommandLineOptions(argc, argv, "MLP Generator");

  MLPGenerator gen(miniBatch, layers, tiles, floatWidth, seed, enableSoftmax,
                   biasAcc);
  return gen.generate(filename);
}
