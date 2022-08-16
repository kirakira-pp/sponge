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
#include "mlir/Support/IndentedOstream.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Transforms/DialectConversion.h"
#include <vector>

using namespace mlir;

class A : llvm::raw_ostream {
	void write_impl(const char *Ptr, size_t Size) override {
		return ;
	}
	uint64_t current_pos() const override {
		return 0;
	}
};

namespace sponge {

class raw_indented_ostream;


struct TestPass : public TestPassBase<TestPass> {
	TestPass() = default;

	// TODO
	void runOnOperation() override;
};


/*  Affine checker: this a test for checking the scf.for is an affine loop or not.
 *  
 *  This test object provide some functions:
 * 	1. isAffine():
 *  2. matchAndRewrite: main logic of this test object.
 */
struct affineChecker : public OpRewritePattern<scf::ForOp> {
	using OpRewritePattern<scf::ForOp>::OpRewritePattern;

	bool isAffine(Value value) const {
		return true;
	}

	// Main logic
	LogicalResult matchAndRewrite(scf::ForOp loop, PatternRewriter &rewriter) const final {
		// Prepare lowerbound, upperbound and step value.
		Value lb = loop.getLowerBound();
		Value ub = loop.getUpperBound();
		Value step = loop.getStep();

		// debug: print the upperbound, lowerbound, and step.
		bool debug = false;
		if(debug) {
			llvm::raw_ostream* raw_os = (llvm::raw_ostream*)new A();
			mlir::raw_indented_ostream* os = new mlir::raw_indented_ostream(*raw_os);

			(*os) << lb << "\n";
		}

		// check upperbound is affine or not. This may use affine helping function.
		if(isAffine(lb) && isAffine(ub) && isAffine(step)) {
			puts("yes");
		}
		else {
			puts("no");
		}

		return success();
	}
};

void TestPass::runOnOperation() {
	RewritePatternSet patterns(&getContext());

	patterns.insert<affineChecker>(&getContext());
}


std::unique_ptr<mlir::Pass> createTestPass() {
  return std::make_unique<TestPass>();
}

} // namespace sponge 

