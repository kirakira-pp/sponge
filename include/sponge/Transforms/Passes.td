//===- Passes.td - sponge pass definition file --------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SPONGE_DIALECT_SPONGE_TRANSFORMS_PASSES
#define SPONGE_DIALECT_SPONGE_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
// include "mlir/IR/OpBase.td"
// include "mlir/IR/BuiltinOps.td"

//===----------------------------------------------------------------------===//
//
// test pass
//
//===----------------------------------------------------------------------===//

def TestPass : Pass<"test-pass"> {
  let summary = "A test pass. Check the scf.for loop can be transformed to affine.for.";
  let constructor = "sponge::createTestPass()";
}

//===----------------------------------------------------------------------===//
//
// liveness test pass
//
//===----------------------------------------------------------------------===//

def LiveTestPass : Pass<"live-test-pass", "::mlir::func::FuncOp"> {
  let summary = "A liveness test pass.";
  let constructor = "sponge::createLiveTestPass()";
}

#endif // SPONGE_DIALECT_SPONGE_TRANSFORMS_PASSES

