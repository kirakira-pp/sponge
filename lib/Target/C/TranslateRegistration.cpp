//===- TranslateRegistration.cpp - Register translation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#include "sponge/Target/C/CEmitter.h"
#include "mlir/Tools/mlir-translate/Translation.h"
// #include "llvm/Support/CommandLine.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// C registration
//===----------------------------------------------------------------------===//

#include "sponge/sponge.h"

namespace sponge {
  void registerToCTranslation() {
    static llvm::cl::opt<bool> declareVariablesAtTop(
        "declare-variables-at-top",
        llvm::cl::desc("Declare variables at top when emitting C/C++"),
        llvm::cl::init(false));

    TranslateFromMLIRRegistration reg(
        "mlir-to-c",
        [](ModuleOp module, raw_ostream &output) {
          return sponge::translateToC(
              module, output,
              /*declareVariablesAtTop=*/declareVariablesAtTop);
        },
        [](DialectRegistry &registry) {
          registry.insert<spongeDialect,
                          func::FuncDialect,
                          arith::ArithmeticDialect,
                          scf::SCFDialect,
                          AffineDialect,
                          math::MathDialect,
                          memref::MemRefDialect>();
        });
  }
} // namespace sponge
