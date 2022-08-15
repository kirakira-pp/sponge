//===- PassDetail.h - sponge Transform Pass class details --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_SPONGE_TRANSFORMS_PASSDETAIL_H
#define DIALECT_SPONGE_TRANSFORMS_PASSDETAIL_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

// namespace mlir {
// class AffineDialect;
// }

namespace sponge {

class spongeDialect;

#define GEN_PASS_CLASSES
#include "sponge/Transforms/Passes.h.inc"

} // namespace sponge

#endif // DIALECT_SPONGE_TRANSFORMS_PASSDETAIL_H

