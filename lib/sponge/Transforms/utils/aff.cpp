#include "sponge/Transforms/utils/aff.h"


void test() {
  puts("aff.cpp");
}

// if '+', '-': lhs and rhs must be affine exprs
// if '*': lhs and rhs must be one const and one affine expr
// if '/', '%': lhs must be affine expr and rhs must be const
// if is const, is affine expr
// if is symbol, is affine expr

bool isConst(Value value) {
  if(auto v = value.getDefiningOp<mlir::arith::ConstantOp>()) return true;
  return false;
}

bool isAffineExpr(Value value) {
  llvm::outs() << value << "\n";

  // +, -
  if(auto v = value.getDefiningOp<mlir::arith::AddIOp>()) {
    return isAffineExpr(v -> getOperand(0)) && isAffineExpr(v.getOperand(1));
  }

  if(auto v = value.getDefiningOp<mlir::arith::SubIOp>()) {
    return isAffineExpr(v.getOperand(0)) && isAffineExpr(v.getOperand(1));
  }

  // *
  if(auto v = value.getDefiningOp<mlir::arith::MulIOp>()) {
    return (isAffineExpr(v.getOperand(0)) && isConst(v.getOperand(1))) ||
         (isConst(v.getOperand(0)) && isAffineExpr(v.getOperand(1)));
  }

  // /, %
  if(auto v = value.getDefiningOp<mlir::arith::DivUIOp>()) {
    return isAffineExpr(v.getOperand(0)) && isConst(v.getOperand(1));
  }
  if(auto v = value.getDefiningOp<mlir::arith::DivSIOp>()) {
    return isAffineExpr(v.getOperand(0)) && isConst(v.getOperand(1));
  }


  if(auto v = value.getDefiningOp<mlir::arith::RemSIOp>()) {
    return isAffineExpr(v.getOperand(0)) && isConst(v.getOperand(1)); 
  }

  // IndexCastOp
  if(auto v = value.getDefiningOp<mlir::arith::IndexCastOp>())
    return isAffineExpr(v.getOperand());

  // symbol
  if(isValidSymbol(value)) return true;

  // const
  if(isConst(value)) return true;

  if(auto v = value.getDefiningOp<mlir::arith::ConstantIndexOp>())
    return true;
  
  // block argument [Warning]
  if (auto ba = value.dyn_cast<BlockArgument>()) {
    auto *owner = ba.getOwner();
    assert(owner);

    auto *parentOp = owner->getParentOp();
    if (!parentOp) {
      owner->dump();
      llvm::errs() << " ba: " << ba << "\n";
    }
    assert(parentOp);
    if (isa<mlir::FunctionOpInterface>(parentOp))
      return true;
    if (auto af = dyn_cast<AffineForOp>(parentOp))
      return af.getInductionVar() == ba;

    // TODO ensure not a reduced var
    if (isa<mlir::AffineParallelOp>(parentOp))
      return true;

    if (isa<mlir::FunctionOpInterface>(parentOp))
      return true;
  }


  return false;
}


void makeAffineBound(scf::ForOp loop, llvm::SmallVector<Value>& ha, AffineMap& mp) {
  return ;
}
