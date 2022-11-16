//===- miniEmitC-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "sponge/sponge.h"
// If not include this, the vtable of sponge::spongeDialect::spongeDialect will be disappear.
// #include "sponge/spongeDialect.cpp.inc"
#include "sponge/Transforms/Passes.h"


int main(int argc, char **argv) {
  // Register Dialect
  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<sponge::spongeDialect>();

  // Register Passes
  mlir::registerAllPasses();
  sponge::registerspongePasses();

  return mlir::asMainReturnCode(
    // Attention: If preloadDialectsInContext is false, the rewriting will failed.
    mlir::MlirOptMain(argc, argv, "sponge optimizer driver\n", registry, 
                      /*preloadDialectsInContext*/true));
}
