//===- Passes.h - sponge Transform Passes ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines transform passes owned by the sponge dialect.
//
//===----------------------------------------------------------------------===//

#ifndef SPONGE_DIALECT_SPONGE_TRANSFORMS_PASSES_H
#define SPONGE_DIALECT_SPONGE_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"

// using namespace mlir;

namespace sponge {

std::unique_ptr<mlir::Pass> createTestPass();

#define GEN_PASS_REGISTRATION
#include "sponge/Transforms/Passes.h.inc"

} // namespace sponge

#endif // SPONGE_DIALECT_SPONGE_TRANSFORMS_PASSES_H

