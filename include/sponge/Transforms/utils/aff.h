#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/FunctionInterfaces.h"
#include <cstdio>

using namespace mlir;

void test();

bool isAffineExpr(Value value);

void makeAffineBound(scf::ForOp loop, llvm::SmallVector<Value>& ha, AffineMap& mp);

