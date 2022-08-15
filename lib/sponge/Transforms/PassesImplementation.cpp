//===- PassesImplementation.cpp - passes implementataion --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "sponge/Transforms/Passes.h"
#include "sponge/sponge.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <vector>

using namespace mlir;

namespace sponge {

struct TestPass : public TestPassBase<TestPass> {
	TestPass() = default;

	// TODO
	void runOnOperation() override;
};

void TestPass::runOnOperation() {}


std::unique_ptr<mlir::Pass> createTestPass() {
  return std::make_unique<TestPass>();
}

} // namespace sponge 

