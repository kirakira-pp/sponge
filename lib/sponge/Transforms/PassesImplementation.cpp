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
#include "sponge/Transforms/utils/aff.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/DialectConversion.h"
#include <vector>

using namespace mlir;

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


	// Warning
	AffineMap getMultiSymbolIdentity(Builder &B, unsigned rank) const {
		SmallVector<AffineExpr, 4> dimExprs;
		dimExprs.reserve(rank);
		for (unsigned i = 0; i < rank; ++i)
		dimExprs.push_back(B.getAffineSymbolExpr(i));
		return AffineMap::get(/*dimCount=*/0, /*symbolCount=*/rank, dimExprs,
							B.getContext());
  	}
	
	LogicalResult convertScfFor(scf::ForOp loop, PatternRewriter &rewriter) const {
		// Prepare lowerbound, upperbound and step value.
		Value lb = loop.getLowerBound();
		Value ub = loop.getUpperBound();
		Value step = loop.getStep();

		// Convert scf loop
		if(isAffineExpr(lb) && isAffineExpr(ub) && isAffineExpr(step)) {
			// loop.emitWarning() << "This op has valid lb, ub and step." << "\n";
			OpBuilder builder(loop);

			// Make affine lowerbound
			SmallVector<Value> lbvec = {loop.getLowerBound()};
			AffineMap lbmp = getMultiSymbolIdentity(builder, lbvec.size());
			// TODO
			

			// Make affine upperbound
			SmallVector<Value> ubvec = {loop.getUpperBound()};
			AffineMap ubmp = getMultiSymbolIdentity(builder, ubvec.size());
			// TODO

			// step
			auto v = loop.getStep().getDefiningOp<arith::ConstantIndexOp>();
			int64_t st = v ? v.value() : 1;

			AffineForOp nloop = rewriter.create<AffineForOp>(loop.getLoc(), lbvec, lbmp, ubvec, ubmp, st, 
										 loop.getIterOperands());

			// block
			SmallVector<Value> vals;
			rewriter.setInsertionPointToStart(&nloop.region().front());
			for (Value arg : nloop.region().front().getArguments()) {
				// if (arg == nloop.getInductionVar()) {
				// arg = rewriter.create<arith::AddIOp>(
				// 	loop.getLoc(), loop.getLowerBound(),
				// 	rewriter.create<arith::MulIOp>(loop.getLoc(), arg, loop.getStep()));
				// }
				vals.push_back(arg);
			}

			rewriter.mergeBlocks(&loop.getRegion().front(),
                           &nloop.region().front(), vals);

			// yeild
			auto y = cast<scf::YieldOp>(nloop.getRegion().front().getTerminator());
			rewriter.setInsertionPoint(y);
			rewriter.create<AffineYieldOp>(y.getLoc(),
											y.getOperands());
			rewriter.eraseOp(y);

			// replace
			rewriter.replaceOp(loop, nloop.getResults());
		}

		

		return success();
	}

	// Main logic
	LogicalResult matchAndRewrite(scf::ForOp loop, PatternRewriter &rewriter) const final {
		if(failed(convertScfFor(loop, rewriter))) return failure();

		return success();
	}
		
};
struct affineChecker2 : public OpRewritePattern<memref::LoadOp> {
	using OpRewritePattern<memref::LoadOp>::OpRewritePattern;
	// LogicalResult convertScfIf(scf::IfOp op, PatternRewriter &rewriter) const {
	// 	return success();
	// }
	// LogicalResult convertStore() const {
	// 	return success();
	// }

	LogicalResult convertLoad(memref::LoadOp loadop, PatternRewriter &rewritter) const {
		puts("[debug] prepare to convert memref::LoadOp");

		// Check all expression of load op
		if (!llvm::all_of(loadop.getIndices(), isAffineExpr))
      		return failure();

		// Convert memref::LoadOp to affine::LoadOp
		// auto nload = write.create<AffineLoadOp>();

		auto memtype = loadop.getMemRef().getType().cast<MemRefType>();
		int64_t rank = memtype.getRank();
		llvm::outs() << "Rank of loadop: " << rank << "\n";

		SmallVector<AffineExpr, 4> dimExprs;
		dimExprs.reserve(rank);
		for (unsigned i = 0; i < rank; ++i)
		dimExprs.push_back(rewritter.getAffineSymbolExpr(i));
		auto map = AffineMap::get(/*dimCount=*/0, /*symbolCount=*/rank, dimExprs,
								  rewritter.getContext());
		llvm::outs() << map << "\n";

		return success();
	}


	// Main logic
	LogicalResult matchAndRewrite(memref::LoadOp loadop, PatternRewriter &rewriter) const final {
		if(failed(convertLoad(loadop, rewriter))) return failure();

		return success();
	}
};

void TestPass::runOnOperation() {
	// Print debug message
	puts("Hi. This is TestPass. This pass will check the scf.for is affine.for or not.");
	puts("If you don't want to see this message please delete the message in lib/sponge/Transforms/PassesImplementation.cpp");
	puts("------------");

	RewritePatternSet patterns(&getContext());

	patterns.insert<affineChecker>(&getContext());
	patterns.insert<affineChecker2>(&getContext());

	GreedyRewriteConfig config;
  	(void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                     config);
}


//===----------------------------------------------------------------------===//
// LiveTestPass
//===----------------------------------------------------------------------===//

struct LiveTestPass : public LiveTestPassBase<LiveTestPass> {
public:
	LiveTestPass() = default;

	void unwarpRegion(mlir::Region&);
	void unwarpBlock(mlir::Block&);
	void test(mlir::arith::ConstantFloatOp&);
	void test2(mlir::AffineForOp&);
	void runOnOperation() override;

private:
	Liveness* li = nullptr;
};

void LiveTestPass::runOnOperation() {
	llvm::outs() << "Start live test pass\n";
	mlir::func::FuncOp func = getOperation();
	llvm::outs() << func.getName() << "\n";
	llvm::outs() << "============================================\n";
	li = new Liveness(getOperation());

	unwarpRegion(getOperation().getRegion());
	/*
	// for(mlir::Block& block : region.getBlocks()) {
	// 	for(auto& op : block) {
	// 		// llvm::outs() << op << '\n';
	// 		// llvm::StringRef opname = op.getName().getStringRef();
	// 		// if(opname.compare("affine.load")) continue;

	// 		// llvm::outs() << "Found a affineload\n";

	// 		mlir::Liveness* ln = new Liveness(&op);
	// 		// for(auto val : op.getOperands()) {
	// 		// 	for(auto& testop : block) {
	// 		// 		llvm::outs() << val << " is dead after " << testop << "\n";
	// 		// 		llvm::outs() << (ln->isDeadAfter(val, &testop) ? "yes\n" : "no\n");
	// 		// 	}
	// 		// }
			
	// 		ln -> dump();
	// 		ln -> print(llvm::outs());
	// 	}
	// }
	*/
}

void LiveTestPass::unwarpRegion(mlir::Region& region) {
	// llvm::outs() << "Start a region\n";
	for(mlir::Block& blk : region) {
		unwarpBlock(blk);
	}
}

void LiveTestPass::unwarpBlock(mlir::Block& block) {
	auto liveInSet = li->getLiveIn(&block);
	auto liveOutSet = li->getLiveOut(&block);
	llvm::outs() << "///////////////////////////////////////////\n";
	llvm::outs() << "live in\n";
	for(auto in : liveInSet) llvm::outs() << in << "\n";
	llvm::outs() << "live out\n";
	for(auto out : liveOutSet) llvm::outs() << out << "\n";
	llvm::outs() << "///////////////////////////////////////////\n";

	// llvm::outs() << "Start a block\n";
	for(mlir::Operation& op : block) {
		llvm::outs() << "============================================\n";
		llvm::outs() << op << ":\n";
		llvm::outs() << "============================================\n";
		auto val = op.getResults();
		llvm::outs() << "  There are " << val.size() << " values in results\n";

		for(auto vv : val) {
			llvm::outs() << "["; vv.print(llvm::outs()); llvm::outs() << "]\n";
			mlir::Liveness::OperationListT list = li->resolveLiveness(vv);

			for(auto element : list) {
				llvm::outs() << "    " << *element << "\n";
			}
			puts("");
		}

		llvm::outs() << "------------------------------------------------\n";
		auto operands = op.getOperands();
		llvm::outs() << "  There are " << operands.size() << " values in operands\n";
		for(auto pp : operands) {
			llvm::outs() << "[" << pp << "]\n";
			mlir::Liveness::OperationListT list = li->resolveLiveness(pp);

			for(auto element : list) {
				llvm::outs() << "    " << *element << "\n";
			}
			puts("");
		}
		if(op.getName().getStringRef().str() ==  "affine.for") {
			llvm::outs() << "------------------------------------------------\n";
			auto iter_args = dyn_cast<mlir::AffineForOp, Operation>(op).getRegionIterArgs();

			for(auto arg : iter_args) {
				llvm::outs() << "[" << arg << "]\n";
				mlir::Liveness::OperationListT list = li->resolveLiveness(arg);

				for(auto element : list) {
					llvm::outs() << "    " << *element << "\n";
				}
				puts("");
			}
		}

		llvm::outs() << "============================================\n\n";

		if(op.getRegions().size() > 0) {
			auto regions = op.getRegions();

			for(auto& region : regions) {
				unwarpRegion(region);
			}
		}

		// llvm::TypeSwitch<Operation*, void>(&op)
		// 	.Case<mlir::arith::ConstantFloatOp>(
		// 		[&](auto op) { return test(op); })
		// 	.Case<mlir::AffineForOp> (
		// 		[&](auto op) { test2(op); })
		// 	.Default([&](Operation *) {});

	}
}

void LiveTestPass::test(mlir::arith::ConstantFloatOp& op) {
	llvm::outs() << "Start a test\n";
	
	llvm::APFloat val = op.value();
	llvm::outs() << val.convertToDouble() << "\n";
	
	mlir::Value tmp = op.getResult();
	mlir::Liveness::OperationListT a = li->resolveLiveness(tmp);
	for(mlir::Operation* aa : a) {
		llvm::outs() << *aa << "\n";
	}

	return ;
}

void LiveTestPass::test2(mlir::AffineForOp& forop) {
	return ;
}


std::unique_ptr<mlir::Pass> createTestPass() { return std::make_unique<TestPass>(); }
std::unique_ptr<mlir::Pass> createLiveTestPass() { return std::make_unique<LiveTestPass>(); }

} // namespace sponge 

