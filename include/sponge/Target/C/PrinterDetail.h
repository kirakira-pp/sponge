#include "CEmitter.h"

using namespace mlir::arith;

// ===----------------------------------------------------------------------===//
// Arithmetic dialect
// ===----------------------------------------------------------------------===//
static LogicalResult printOperation(CEmitter &emitter, ConstantOp constantOp) {
	Operation* operation = constantOp.getOperation();
	OpResult result = operation->getResult(0);
	Attribute value = constantOp.getValue();

	// Only emit an assignment as the variable was already declared when printing
	// the FuncOp.
	if (emitter.shouldDeclareVariablesAtTop()) {
		if (failed(emitter.emitVariableAssignment(result))) return failure();
		return emitter.emitAttribute(operation->getLoc(), value);
	}

	// Emit a variable declaration.
	if (failed(emitter.emitAssignPrefix(*operation)))
		return failure();

	emitter.createContextMapping(emitter.getOrCreateName(operation -> getResult(0)), 
								 emitter.typeAttrToString(constantOp.getLoc(), value));

	return emitter.emitAttribute(operation->getLoc(), value);
}

static LogicalResult printOperation(CEmitter &emitter, SelectOp selectop) {
	Operation *operation = selectop.getOperation();
	if (failed(emitter.emitAssignPrefix(*operation)))
		return failure();

	raw_ostream &os = emitter.ostream();

	// "cond" ? "valA" : "valB"
	StringRef cond = emitter.getContextMapping(emitter.getOrCreateName(selectop.getOperand(0)));
	StringRef valA = emitter.getContextMapping(emitter.getOrCreateName(selectop.getOperand(1)));
	StringRef valB = emitter.getContextMapping(emitter.getOrCreateName(selectop.getOperand(2)));
	std::string tmp = std::string(cond.data()) + " ? " + 
					  std::string(valA.data()) + " : " + std::string(valB.data());
	os << tmp;

	// Remove scalar dependences
	emitter.createContextMapping(emitter.getOrCreateName(selectop.getResult()), "(" + tmp + ")");

	return success();
}

static LogicalResult printOperation(CEmitter &emitter, MulFOp mulfop) {
	Operation *operation = mulfop.getOperation();
	if (failed(emitter.emitAssignPrefix(*operation)))
		return failure();

	raw_ostream &os = emitter.ostream();

	// "res" = "opA" * "opB"
	StringRef res = emitter.getOrCreateName(operation -> getResult(0));
	StringRef opA = emitter.getOrCreateName(mulfop.getOperand(0));
	StringRef opB = emitter.getOrCreateName(mulfop.getOperand(1));

	// Print the result
	std::string tmp = std::string(emitter.getContextMapping(opA).data()) + " * " + 
					  std::string(emitter.getContextMapping(opB).data());
	os << tmp;

	// Remove scalar dependences (DeLICM simply)
	emitter.createContextMapping(res, "(" + tmp + ")");
	std::set<std::string> s;
	std::set<std::string> s1 = emitter.getKill(opA.data());
	std::set<std::string> s2 = emitter.getKill(opB.data());
	std::set_union(s1.begin(), s1.end(), s2.begin(), s2.end(), std::inserter(s, s.begin()));
	emitter.updateKill(res.data(), s);
	
	return success();
}

static LogicalResult printOperation(CEmitter &emitter, MulIOp muliop) {
	raw_ostream &os = emitter.ostream();

	Operation *operation = muliop.getOperation();
	if (failed(emitter.emitAssignPrefix(*operation)))
		return failure();

	StringRef res = emitter.getOrCreateName(operation -> getResult(0));
	StringRef opA = emitter.getOrCreateName(muliop.getOperand(0));
	StringRef opB = emitter.getOrCreateName(muliop.getOperand(1));

	std::string tmp = std::string(emitter.getContextMapping(opA).data()) + " * " + std::string(emitter.getContextMapping(opB).data());
	os << tmp;
	emitter.createContextMapping(res, "(" + tmp + ")");

	// os << emitter.getOrCreateName(muliop.getOperand(0)) << " * "
	//    << emitter.getOrCreateName(muliop.getOperand(1));
	return success();
}

static LogicalResult printOperation(CEmitter &emitter, AddFOp addfop) {
	raw_ostream &os = emitter.ostream();

	Operation *operation = addfop.getOperation();  

	if (failed(emitter.emitAssignPrefix(*operation)))
		return failure();

	StringRef res = emitter.getOrCreateName(operation -> getResult(0));
	StringRef opA = emitter.getOrCreateName(addfop.getOperand(0));
	StringRef opB = emitter.getOrCreateName(addfop.getOperand(1));

	std::string tmp = std::string(emitter.getContextMapping(opA).data()) + " + " + 
					  std::string(emitter.getContextMapping(opB).data());
	os << tmp;
	emitter.createContextMapping(res, "(" + tmp + ")");

	std::set<std::string> s;
	std::set<std::string> s1 = emitter.getKill(opA.data());
	std::set<std::string> s2 = emitter.getKill(opB.data());
	std::set_union(s1.begin(), s1.end(), s2.begin(), s2.end(), std::inserter(s, s.begin()));
	// os << emitter.getContextMapping(res);

	// llvm::outs() << "\n------------------------\n";
	// llvm::outs() << "Addfop operand use info: \n";
	// const auto& ur = addfop.getResult().getUses();
	// llvm::outs() << "  " << emitter.getOrCreateName(addfop.getResult()) << ":\n";
	// for(auto urit = ur.begin(); urit != ur.end(); urit++) {
	// 	llvm::outs() << "    " << urit -> getOwner() -> getName() << " at " << urit ->getOwner() -> getLoc() << "\n";
	// }

	// const auto& usr = addfop.getOperand(1).getUsers();
	// llvm::outs() << "  " << emitter.getOrCreateName(addfop.getOperand(1)) << ":\n";
	// for(auto usrit = usr.begin(); usrit != usr.end(); usrit++) {
	// 	llvm::outs() << "    " << usrit -> getName() << " at " << usrit -> getLoc() << "\n";
	// }


	// llvm::outs() << "------------------------\n";


	emitter.updateKill(res.data(), s);

	return success();
	}

static LogicalResult printOperation(CEmitter &emitter, AddIOp addiop) {
	raw_ostream &os = emitter.ostream();

	Operation *operation = addiop.getOperation();
	
	if (failed(emitter.emitAssignPrefix(*operation)))
		return failure();
	
	StringRef res = emitter.getOrCreateName(operation -> getResult(0));
	StringRef opA = emitter.getContextMapping(emitter.getOrCreateName(addiop.getOperand(0)));
	StringRef opB = emitter.getContextMapping(emitter.getOrCreateName(addiop.getOperand(1)));
	std::string tmp = std::string(emitter.getContextMapping(opA).data()) + " + " + 
						std::string(emitter.getContextMapping(opB).data());
	os << tmp;
	emitter.createContextMapping(res, "(" + tmp + ")");
	
	// os << emitter.getContextMapping(res);

	return success();
}

static LogicalResult printOperation(CEmitter &emitter, SubIOp subiop) {
	raw_ostream &os = emitter.ostream();

	Operation *operation = subiop.getOperation();
	if (failed(emitter.emitAssignPrefix(*operation)))
		return failure();
	StringRef res = emitter.getOrCreateName(operation -> getResult(0));
	StringRef opA = emitter.getOrCreateName(subiop.getOperand(0));
	StringRef opB = emitter.getOrCreateName(subiop.getOperand(1));

	std::string tmp = std::string(emitter.getContextMapping(opA).data()) + " - " + std::string(emitter.getContextMapping(opB).data());
	os << tmp;
	emitter.createContextMapping(res, "(" + tmp + ")");

	// os << emitter.getOrCreateName(subiop.getOperand(0)) << " - "
	//    << emitter.getOrCreateName(subiop.getOperand(1));
	return success();
}

static LogicalResult printOperation(CEmitter &emitter, SubFOp subfop) {
	raw_ostream &os = emitter.ostream();

	Operation *operation = subfop.getOperation();
	if (failed(emitter.emitAssignPrefix(*operation)))
		return failure();
	StringRef res = emitter.getOrCreateName(operation -> getResult(0));
	StringRef opA = emitter.getOrCreateName(subfop.getOperand(0));
	StringRef opB = emitter.getOrCreateName(subfop.getOperand(1));

	std::string tmp = std::string(emitter.getContextMapping(opA).data()) + " - " + 
						std::string(emitter.getContextMapping(opB).data());
	os << tmp;
	emitter.createContextMapping(res, "(" + tmp + ")");

	std::set<std::string> s;
	std::set<std::string> s1 = emitter.getKill(opA.data());
	std::set<std::string> s2 = emitter.getKill(opB.data());
	std::set_union(s1.begin(), s1.end(), s2.begin(), s2.end(), std::inserter(s, s.begin()));

	// os << "/*" << "union set of " << opA << " and " << opB << ": ";
	// for(auto iter : s) os << " " << iter;
	// os << "*/";

	emitter.updateKill(res.data(), s);

	// os << emitter.getOrCreateName(subfop.getOperand(0)) << " - "
	//    << emitter.getOrCreateName(subfop.getOperand(1));
	return success();
}

static LogicalResult printOperation(CEmitter &emitter, IndexCastOp indexcastop) {
	raw_ostream &os = emitter.ostream();

	Operation* operation = indexcastop.getOperation();
	std::string tmp;

	if (failed(emitter.emitAssignPrefix(*operation)))
		return failure();

	// os << "(";

	OpResult result = operation -> getResult(0);
	// tmp += "(" + emitter.getTypeStr(result.getOwner() -> getLoc(), result.getType()) + ")";
	tmp += emitter.getContextMapping(emitter.getOrCreateName(indexcastop.getOperand()));

	// if (failed(emitter.emitType(result.getOwner()->getLoc(), result.getType())))
	//   return failure();
	// os << ")" << emitter.getOrCreateName(indexcastop.getOperand());
	os << tmp;
	// os << "// " << emitter.getOrCreateName(result) << " = " << tmp;
	// os << "// map[" << emitter.getOrCreateName(result) << "] = " << tmp;
	emitter.createContextMapping(emitter.getOrCreateName(result), "(" + tmp + ")");
	// os << "// " << emitter.getOrCreateName(result) << " = " << emitter.getContextMapping(emitter.getOrCreateName(result));

	return success();
}

static LogicalResult printOperation(CEmitter &emitter, SIToFPOp sitofpop) {
	raw_ostream &os = emitter.ostream();

	Operation *op = sitofpop.getOperation();

	if (failed(emitter.emitAssignPrefix(*op)))
		return failure();

	std::string tmp;
	// os << "(";
	// OpResult result = op->getResult(0);
	// tmp += "(" + emitter.typeAttrToString(result.getOwner() -> getLoc(), result.getType()) + ")";
	tmp += emitter.getContextMapping(emitter.getOrCreateName(sitofpop.getOperand())).data();

	// if (failed(emitter.emitType(result.getOwner()->getLoc(), result.getType())))
	//   return failure();
	// os << ")" << emitter.getOrCreateName(sitofpop.getOperand());
	os << tmp;
	emitter.createContextMapping(emitter.getOrCreateName(op -> getResult(0)), "(" + tmp + ")");

	return success();
}

static LogicalResult printOperation(CEmitter &emitter, UIToFPOp uitofpop) {
  raw_ostream &os = emitter.ostream();

  Operation *op = uitofpop.getOperation();

  if (failed(emitter.emitAssignPrefix(*op)))
    return failure();

  std::string tmp;
  // os << "(";
//   OpResult result = op->getResult(0);
  // tmp += "(" + emitter.getTypeStr(result.getOwner() -> getLoc(), result.getType()) + ")";
  tmp += emitter.getContextMapping(emitter.getOrCreateName(uitofpop.getOperand())).data();

  // if (failed(emitter.emitType(result.getOwner()->getLoc(), result.getType())))
  //   return failure();
  // os << ")" << emitter.getOrCreateName(sitofpop.getOperand());
  os << tmp;
  emitter.createContextMapping(emitter.getOrCreateName(op -> getResult(0)), "(" + tmp + ")");

  return success();
}

static LogicalResult printOperation(CEmitter &emitter, FPToUIOp fptouiop) {
  raw_ostream &os = emitter.ostream();

  Operation *op = fptouiop.getOperation();

  if (failed(emitter.emitAssignPrefix(*op)))
    return failure();

  std::string tmp;
  // os << "(";
//   OpResult result = op->getResult(0);
  // tmp += "(" + emitter.getTypeStr(result.getOwner() -> getLoc(), result.getType()) + ")";
  tmp += emitter.getContextMapping(emitter.getOrCreateName(fptouiop.getOperand())).data();

  // if (failed(emitter.emitType(result.getOwner()->getLoc(), result.getType())))
  //   return failure();
  // os << ")" << emitter.getOrCreateName(sitofpop.getOperand());
  os << tmp;
  emitter.createContextMapping(emitter.getOrCreateName(op -> getResult(0)), "(" + tmp + ")");

  return success();
}

static LogicalResult printOperation(CEmitter &emitter, FPToSIOp fptosiop) {
  raw_ostream &os = emitter.ostream();

  Operation *op = fptosiop.getOperation();

  if (failed(emitter.emitAssignPrefix(*op)))
    return failure();

  std::string tmp;
  // os << "(";
//   OpResult result = op->getResult(0);
//   tmp += "(" + emitter.typeAttrToString(result.getOwner() -> getLoc(), result.getType()) + ")";
  tmp += emitter.getContextMapping(emitter.getOrCreateName(fptosiop.getOperand()));
  // if (failed(emitter.emitType(result.getOwner()->getLoc(), result.getType())))
  //   return failure();
  // os << ")" << emitter.getOrCreateName(fptosiop.getOperand());

  emitter.createContextMapping(emitter.getOrCreateName(op -> getResult(0)), "(" + tmp + ")");
  os << tmp;

  return success();
}

static LogicalResult printOperation(CEmitter &emitter, ExtSIOp extsiop) {
  raw_ostream &os = emitter.ostream();

  Operation *op = extsiop.getOperation();

  if (failed(emitter.emitAssignPrefix(*op)))
    return failure();

  std::string tmp;
  // os << "(";
//   OpResult result = op->getResult(0);
  // tmp += "(" + emitter.getTypeStr(result.getOwner() -> getLoc(), result.getType()) + ")";
  tmp += emitter.getContextMapping(emitter.getOrCreateName(extsiop.getOperand()));

  emitter.createContextMapping(emitter.getOrCreateName(op -> getResult(0)), "(" + tmp + ")");
  os << tmp;

  return success();
}

static LogicalResult printOperation(CEmitter &emitter, ExtFOp extfop) {
  raw_ostream &os = emitter.ostream();

  Operation *op = extfop.getOperation();

  if (failed(emitter.emitAssignPrefix(*op)))
    return failure();
  os << "(";
  OpResult result = op->getResult(0);
  if (failed(emitter.emitType(result.getOwner()->getLoc(), result.getType())))
    return failure();
  os << ")" << emitter.getOrCreateName(extfop.getOperand());

  return success();
}

static LogicalResult printOperation(CEmitter &emitter, TruncFOp truncfop) {
  raw_ostream &os = emitter.ostream();

  Operation *op = truncfop.getOperation();

  if (failed(emitter.emitAssignPrefix(*op)))
    return failure();
  os << "(";
  OpResult result = op->getResult(0);
  if (failed(emitter.emitType(result.getOwner()->getLoc(), result.getType())))
    return failure();
  os << ")" << emitter.getOrCreateName(truncfop.getOperand());

  return success();
}

static LogicalResult printOperation(CEmitter &emitter, TruncIOp truncfop) {
  raw_ostream &os = emitter.ostream();

  Operation *op = truncfop.getOperation();

  if (failed(emitter.emitAssignPrefix(*op)))
    return failure();
  os << "(";
  OpResult result = op->getResult(0);
  if (failed(emitter.emitType(result.getOwner()->getLoc(), result.getType())))
    return failure();
  os << ")" << emitter.getOrCreateName(truncfop.getOperand());

  return success();
}

static LogicalResult printOperation(CEmitter &emitter, NegFOp negfop) {
  raw_ostream &os = emitter.ostream();

  Operation *op = negfop.getOperation();

  if (failed(emitter.emitAssignPrefix(*op)))
    return failure();
  std::string tmp("(-");
  // os << "(-";
  tmp += std::string(emitter.getContextMapping(emitter.getOrCreateName(negfop.getOperand()))) + ")";
  os << tmp;
  // os << emitter.getOrCreateName(negfop.getOperand()) << ")";

  emitter.createContextMapping(emitter.getOrCreateName(op -> getResult(0)), "(" + tmp + ")");

  // os << "/* Find " << 
  std::set<std::string> s = emitter.getKill( emitter.getOrCreateName(negfop.getOperand()).data() );
  // os << "/* " << emitter.getOrCreateName(op -> getResult(0)).data() << " : ";
  // for(auto iter = s.begin(); iter != s.end(); iter++) os << *iter;
  // os << " */";
  emitter.updateKill(emitter.getOrCreateName(op -> getResult(0)).data(), s);

  return success();
}

static LogicalResult printOperation(CEmitter &emitter, CmpFOp cmpfop) {
  raw_ostream &os = emitter.ostream();

  Operation *op = cmpfop.getOperation();
  if (failed(emitter.emitAssignPrefix(*op)))
    return failure();

  std::string LHS = emitter.getContextMapping(emitter.getOrCreateName(cmpfop.getOperand(0))).data();
  std::string sign;
  std::string RHS = emitter.getContextMapping(emitter.getOrCreateName(cmpfop.getOperand(1))).data();
  // os << "(" << emitter.getOrCreateName(cmpfop.getOperand(0));

  switch (cmpfop.getPredicate()) {
  case CmpFPredicate::OEQ:
  case CmpFPredicate::UEQ:
    // case CmpIPredicate::eq:
    sign = " == ";
	// os << " == ";
    break;
  case CmpFPredicate::OGT:
  case CmpFPredicate::UGT:
    // case CmpIPredicate::sgt:
    // case CmpIPredicate::ugt:
    sign = " > ";
	// os << " > ";
    break;
  case CmpFPredicate::OGE:
  case CmpFPredicate::UGE:
    // case CmpIPredicate::sge:
    // case CmpIPredicate::uge:
    sign = " >= ";
	// os << " >= ";
    break;
  case CmpFPredicate::OLT:
  case CmpFPredicate::ULT:
    // case CmpIPredicate::slt:
    // case CmpIPredicate::ult:
    sign = " < ";
	// os << " < ";
    break;
  case CmpFPredicate::OLE:
  case CmpFPredicate::ULE:
    // case CmpIPredicate::sle:
    // case CmpIPredicate::sle:
    sign = " <= ";
	// os << " <= ";
    break;
  case CmpFPredicate::ONE:
  case CmpFPredicate::UNE:
    // case CmpIPredicate::le:
   	sign = " != ";
	// os << " != ";
    break;
  case CmpFPredicate::AlwaysFalse:
    sign = " False ";
	// os << " False ";
    break;
  case CmpFPredicate::AlwaysTrue:
    sign = " True ";
	// os << " True ";
    break;
  default:
  	sign = " ??? ";
    // os << " ??? ";
    break;
  }
  
  // os << emitter.getOrCreateName(cmpfop.getOperand(1)) << ")";
  os << LHS + sign + RHS;
  emitter.createContextMapping(emitter.getOrCreateName(cmpfop.getResult()), "(" + LHS + sign + RHS + ")");

  return success();
}

static LogicalResult printOperation(CEmitter &emitter, CmpIOp cmpiop) {
  raw_ostream &os = emitter.ostream();

  Operation *op = cmpiop.getOperation();
  if (failed(emitter.emitAssignPrefix(*op)))
    return failure();
  std::string LHS(emitter.getContextMapping(emitter.getOrCreateName(cmpiop.getOperand(0))));
  std::string sign;
  std::string RHS(emitter.getContextMapping(emitter.getOrCreateName(cmpiop.getOperand(1))));

  // os << "(" << emitter.getOrCreateName(cmpiop.getOperand(0));

  switch (cmpiop.getPredicate()) {
  case CmpIPredicate::eq:
  	sign = " == ";
    // os << " == ";
    break;
  case CmpIPredicate::sgt:
  case CmpIPredicate::ugt:
  	sign = " > ";
    // os << " > ";
    break;
  case CmpIPredicate::sge:
  case CmpIPredicate::uge:
  	sign = " >= ";
    // os << " >= ";
    break;
  case CmpIPredicate::slt:
  case CmpIPredicate::ult:
    sign = " < ";
	// os << " < ";
    break;
  case CmpIPredicate::sle:
  case CmpIPredicate::ule:
    sign = " <= ";
	// os << " <= ";
    break;
  case CmpIPredicate::ne:
    sign = " != ";
	// os << " != ";
    break;
  }

  // os << emitter.getOrCreateName(cmpiop.getOperand(1)) << ")";
  os << LHS + sign + RHS;

  emitter.createContextMapping(emitter.getOrCreateName(cmpiop.getResult()), "(" + LHS + sign + RHS + ")");

  return success();
}

static LogicalResult printOperation(CEmitter &emitter, DivFOp divfop) {
  raw_ostream &os = emitter.ostream();

  Operation *operation = divfop.getOperation();
  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  StringRef res = emitter.getOrCreateName(operation -> getResult(0));
  StringRef opA = emitter.getOrCreateName(divfop.getOperand(0));
  StringRef opB = emitter.getOrCreateName(divfop.getOperand(1));

  std::string tmp = std::string(emitter.getContextMapping(opA).data()) + " / " + 
                    std::string(emitter.getContextMapping(opB).data());
  os << tmp;
  emitter.createContextMapping(res, "(" + tmp + ")");

  std::set<std::string> s;
  std::set<std::string> s1 = emitter.getKill(opA.data());
  std::set<std::string> s2 = emitter.getKill(opB.data());
  std::set_union(s1.begin(), s1.end(), s2.begin(), s2.end(), std::inserter(s, s.begin()));
  
  // os << "/*" << "union set of " << opA << " and " << opB << ": ";
  // for(auto iter : s) os << " " << iter;
  // os << "*/";

  emitter.updateKill(res.data(), s);

  // os << emitter.getOrCreateName(divfop.getOperand(0)) << " / "
  //    << emitter.getOrCreateName(divfop.getOperand(1));

  return success();
}

static LogicalResult printOperation(CEmitter &emitter, DivSIOp divsiop) {
  raw_ostream &os = emitter.ostream();

  Operation *operation = divsiop.getOperation();
  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  StringRef res = emitter.getOrCreateName(operation -> getResult(0));
  StringRef opA = emitter.getOrCreateName(divsiop.getOperand(0));
  StringRef opB = emitter.getOrCreateName(divsiop.getOperand(1));

  std::string tmp = std::string(emitter.getContextMapping(opA).data()) + " / " + std::string(emitter.getContextMapping(opB).data());
  os << tmp;
  emitter.createContextMapping(res, "(" + tmp + ")");

  // os << emitter.getOrCreateName(divsiop.getOperand(0)) << " / "
  //    << emitter.getOrCreateName(divsiop.getOperand(1));

  return success();
}

static LogicalResult printOperation(CEmitter &emitter, RemSIOp remsiop) {
  raw_ostream &os = emitter.ostream();

  Operation *operation = remsiop.getOperation();
  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  StringRef res = emitter.getOrCreateName(operation -> getResult(0));
  StringRef opA = emitter.getOrCreateName(remsiop.getOperand(0));
  StringRef opB = emitter.getOrCreateName(remsiop.getOperand(1));

  std::string tmp = std::string(emitter.getContextMapping(opA).data()) + " % " + std::string(emitter.getContextMapping(opB).data());
  os << tmp;
  emitter.createContextMapping(res, "(" + tmp + ")");

  // os << emitter.getOrCreateName(divsiop.getOperand(0)) << " / "
  //    << emitter.getOrCreateName(divsiop.getOperand(1));

  return success();
}

//===----------------------------------------------------------------------===//
// Func dialect
//===----------------------------------------------------------------------===//

static LogicalResult printOperation(CEmitter &emitter, func::CallOp callOp) {
	if (failed(emitter.emitAssignPrefix(*callOp.getOperation())))
		return failure();

	raw_ostream &os = emitter.ostream();
	os << callOp.getCallee() << "(";
	if (failed(emitter.emitOperands(*callOp.getOperation())))
		return failure();
	os << ")";
	return success();
}

static LogicalResult printOperation(CEmitter &emitter, func::ReturnOp returnOp) {
  raw_ostream &os = emitter.ostream();
  os << "return";
  switch (returnOp.getNumOperands()) {
  case 0:
    return success();
  case 1:
    os << " " << emitter.getOrCreateName(returnOp.getOperand(0));
    return success(emitter.hasValueInScope(returnOp.getOperand(0)));
  default:

	returnOp.emitError("Cannot return more than one value in C.");
    // os << " std::make_tuple(";
    // if (failed(emitter.emitOperandsAndAttributes(*returnOp.getOperation())))
    //   return failure();
    // os << ")";
    return success();
  }
}

static LogicalResult printOperation(CEmitter &emitter, func::FuncOp functionOp) {
	// We need to declare variables at top if the function has multiple blocks.
	if (!emitter.shouldDeclareVariablesAtTop() &&
		functionOp.getBlocks().size() > 1) {
		return functionOp.emitOpError(
			"with multiple blocks needs variables declared at top");
		}

	CEmitter::Scope scope(emitter);

	raw_indented_ostream &os = emitter.ostream();
	if (failed(emitter.emitTypes(functionOp.getLoc(),
							     functionOp.getFunctionType().getResults())))
	return failure();

  	os << " " << functionOp.getName();
 	os << "(";

 	if (failed(interleaveCommaWithError(
		functionOp.getArguments(), os,
		[&](BlockArgument arg) -> LogicalResult {
			// Memref type has dimensions
			if (auto tType = arg.getType().dyn_cast<MemRefType>()) {
				// Print dimensions
				if (failed(emitter.emitType(functionOp.getLoc(), arg.getType(),
						   emitter.getOrCreateName(arg))))
					return failure();
			}
			// Other types
			else {
				if (failed(emitter.emitType(functionOp.getLoc(), arg.getType())))
					return failure();
					
				os << " " << emitter.getOrCreateName(arg);
			}
			return success();
		}
	)))
	return failure();
  	os << ") {\n";
	os.indent();
	if (emitter.shouldDeclareVariablesAtTop()) {
		// Declare all variables that hold op results including those from nested
		// regions.
		WalkResult result =
			functionOp.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
				for (OpResult result : op->getResults()) {
					if (failed(emitter.emitVariableDeclaration(
							   result, /*trailingSemicolon=*/true))) {
						return WalkResult(
							op->emitError("unable to declare result variable for op"));
					}
				}
				return WalkResult::advance();
			});
		if (result.wasInterrupted()) return failure();
	}

	Region::BlockListType &blocks = functionOp.getBlocks();
	// Create label names for basic blocks.
	for (Block &block : blocks) {
		emitter.getOrCreateName(block);
	}

	// Declare variables for basic block arguments.
	for (auto it = std::next(blocks.begin()); it != blocks.end(); ++it) {
		Block &block = *it;
		for (BlockArgument &arg : block.getArguments()) {
			if (emitter.hasValueInScope(arg))
				return functionOp.emitOpError(" block argument #")
					   << arg.getArgNumber() << " is out of scope";
			if (failed(emitter.emitType(block.getParentOp()->getLoc(), arg.getType()))) {
				return failure();
			}
			os << " " << emitter.getOrCreateName(arg) << ";\n";
		}
	}

	for (Block &block : blocks) {
		// Only print a label if there is more than one block.
		if (blocks.size() > 1) {
			if (failed(emitter.emitLabel(block))) return failure();
		}
		for (Operation &op : block.getOperations()) {
			// When generating code for an scf.if or std.cond_br op no semicolon needs
			// to be printed after the closing brace.
			// When generating code for an scf.for op, printing a trailing semicolon
			// is handled within the printOperation function.
			bool trailingSemicolon = 
				!isa<scf::IfOp, scf::ForOp,			// scf
					 cf::CondBranchOp,				// cf
					 AffineIfOp, AffineForOp>(op);  // affine

			if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/trailingSemicolon)))
				return failure();
		}
	}
	os.unindent() << "}\n";
	return success();
}

//===----------------------------------------------------------------------===//
// MemRef dialect
//===----------------------------------------------------------------------===//

static LogicalResult printOperation(CEmitter &emitter,
                                    memref::AllocaOp allocaop) {
  Operation *op = allocaop.getOperation();

  OpResult result = op->getResult(0);
  if (failed(emitter.emitType(result.getOwner()->getLoc(), result.getType(),
                               emitter.getOrCreateName(allocaop))))
    return failure();

  return success();
}

static LogicalResult printOperation(CEmitter &emitter,
                                    memref::LoadOp memrefloadop) {
  // raw_ostream &os = emitter.ostream();

  Operation *operation = memrefloadop.getOperation();
  // if (failed(emitter.emitAssignPrefix(*operation)))
  //   return failure();

  std::string tmp(emitter.getContextMapping(emitter.getOrCreateName(memrefloadop.getMemRef())).data());
  // os << emitter.getOrCreateName(memrefloadop.getMemRef());

  for (auto i : memrefloadop.getIndices())
    tmp += "[" + std::string(emitter.getOrCreateName(i).data()) + "]";
    // os << "[" << emitter.getOrCreateName(i) << "]";

  // os << tmp;
  emitter.createContextMapping(emitter.getOrCreateName(operation -> getResult(0)), "(" + tmp + ")");

  return success();
}

static LogicalResult printOperation(CEmitter &emitter,
                                    memref::StoreOp memrefstoreop) {
  raw_ostream &os = emitter.ostream();

  std::string LHS(emitter.getContextMapping(emitter.getOrCreateName(memrefstoreop.getMemRef())).data());
  // os << LHS;
  // os << emitter.getOrCreateName(memrefstoreop.getMemRef());
  for (auto i : memrefstoreop.getIndices())
    LHS += "[" + std::string(emitter.getOrCreateName(i).data()) + "]";
    // os << "[" << emitter.getOrCreateName(i) << "]";
    // os << "[" << emitter.getContextMapping(emitter.getOrCreateName(i)) << "]";

  os << "/**/" << LHS;

  std::string tmp(emitter.getContextMapping(emitter.getOrCreateName(memrefstoreop.getValueToStore())));
  os << " = " << tmp;

  emitter.createContextMapping(LHS, tmp);

  return success();
}

//===----------------------------------------------------------------------===//
// Math dialect
//===----------------------------------------------------------------------===//
static LogicalResult printOperation(CEmitter &emitter, math::AbsOp absop) {
  raw_ostream &os = emitter.ostream();

  Operation *op = absop.getOperation();
  if (failed(emitter.emitAssignPrefix(*op)))
    return failure();

  // using math.h fabsf()
  os << "fabsf(" << emitter.getOrCreateName(absop.getOperand()) << ")";

  return success();
}

static LogicalResult printOperation(CEmitter &emitter, math::SqrtOp sqrtop) {
  raw_ostream &os = emitter.ostream();

  Operation *operation = sqrtop.getOperation();
  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

//   StringRef res = emitter.getOrCreateName(operation -> getResult(0));
  StringRef opA = emitter.getOrCreateName(sqrtop.getOperand());

  // std::string tmp(std::string("sqrt(" + emitter.getContextMapping(opA.data()).str() + ")"));
  // os << tmp;
  // emitter.createContextMapping(res, "(" + tmp + ")");

  os << "sqrt(" << emitter.getContextMapping(opA.data()) << ")";

  return success();
}

static LogicalResult printOperation(CEmitter &emitter, math::ExpOp expop) {
  raw_ostream &os = emitter.ostream();

  Operation *op = expop.getOperation();
  if (failed(emitter.emitAssignPrefix(*op)))
    return failure();

  // std::string tmp;
  // tmp += "exp(" + std::string(emitter.getContextMapping(emitter.getOrCreateName(expop.getOperand()))) + ")";

  // os << tmp;
  // os << "exp(" << emitter.getOrCreateName(expop.getOperand()) << ")";
  os << "exp(" << emitter.getContextMapping(emitter.getOrCreateName(expop.getOperand())) << ")";

  // emitter.createContextMapping(emitter.getOrCreateName(op -> getResult(0)), tmp);

  return success();
}

static LogicalResult printOperation(CEmitter &emitter, math::PowFOp powfop) {
  raw_ostream &os = emitter.ostream();

  Operation *op = powfop.getOperation();
  if (failed(emitter.emitAssignPrefix(*op)))
    return failure();

  // std::string tmp;
  // tmp += "powf(" + std::string(emitter.getContextMapping(emitter.getOrCreateName(powfop.getOperand(0)))) + ", " +
          // std::string(emitter.getContextMapping(emitter.getOrCreateName(powfop.getOperand(1)))) + ")";

  os << "powf(" << emitter.getContextMapping(emitter.getOrCreateName(powfop.getOperand(0))) << ", "
     << emitter.getContextMapping(emitter.getOrCreateName(powfop.getOperand(1))) << ")";

  // os << tmp;

  // emitter.createContextMapping(emitter.getOrCreateName(op -> getResult(0)), tmp);

  return success();
}

//===----------------------------------------------------------------------===//
// scf dialect
//===----------------------------------------------------------------------===//

static LogicalResult printOperation(CEmitter &emitter, scf::ForOp forOp) {
  raw_indented_ostream &os = emitter.ostream();

  OperandRange operands = forOp.getIterOperands();
  Block::BlockArgListType iterArgs = forOp.getRegionIterArgs();
  Operation::result_range results = forOp.getResults();

  if (!emitter.shouldDeclareVariablesAtTop()) {
    for (OpResult result : results) {
      if (failed(emitter.emitVariableDeclaration(result,
                                                 /*trailingSemicolon=*/true)))
        return failure();
    }
  }

  for (auto pair : llvm::zip(iterArgs, operands)) {
    if (failed(emitter.emitType(forOp.getLoc(), std::get<0>(pair).getType())))
      return failure();
    os << " " << emitter.getOrCreateName(std::get<0>(pair)) << " = ";
    os << emitter.getOrCreateName(std::get<1>(pair)) << ";";
    os << "\n";
  }

  os << "for (";
  if (failed(
          emitter.emitType(forOp.getLoc(), forOp.getInductionVar().getType())))
    return failure();
  os << " ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " = ";
  os << emitter.getOrCreateName(forOp.getLowerBound());
  os << "; ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " < ";
  os << emitter.getOrCreateName(forOp.getUpperBound());
  os << "; ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " += ";
  os << emitter.getOrCreateName(forOp.getStep());
  os << ") {\n";
  os.indent();

  Region &forRegion = forOp.getRegion();
  auto regionOps = forRegion.getOps();

  // We skip the trailing yield op because this updates the result variables
  // of the for op in the generated code. Instead we update the iterArgs at
  // the end of a loop iteration and set the result variables after the for
  // loop.
  for (auto it = regionOps.begin(); std::next(it) != regionOps.end(); ++it) {
    if (failed(emitter.emitOperation(*it, /*trailingSemicolon=*/true)))
      return failure();
  }

  Operation *yieldOp = forRegion.getBlocks().front().getTerminator();
  // Copy yield operands into iterArgs at the end of a loop iteration.
  for (auto pair : llvm::zip(iterArgs, yieldOp->getOperands())) {
    BlockArgument iterArg = std::get<0>(pair);
    Value operand = std::get<1>(pair);
    os << emitter.getOrCreateName(iterArg) << " = "
       << emitter.getOrCreateName(operand) << ";;\n";
  }

  os.unindent() << "}";

  // Copy iterArgs into results after the for loop.
  for (auto pair : llvm::zip(results, iterArgs)) {
    OpResult result = std::get<0>(pair);
    BlockArgument iterArg = std::get<1>(pair);
    os << "\n"
       << emitter.getOrCreateName(result) << " = "
       << emitter.getOrCreateName(iterArg) << ";;;";
  }

  return success();
}

static LogicalResult printOperation(CEmitter &emitter, scf::IfOp ifOp) {
  raw_indented_ostream &os = emitter.ostream();

  if (!emitter.shouldDeclareVariablesAtTop()) {
    for (OpResult result : ifOp.getResults()) {
      if (failed(emitter.emitVariableDeclaration(result,
                                                 /*trailingSemicolon=*/true)))
        return failure();
    }
  }

  os << "if (";
  if (failed(emitter.emitOperands(*ifOp.getOperation())))
    return failure();
  os << ") {\n";
  os.indent();

  Region &thenRegion = ifOp.getThenRegion();
  for (Operation &op : thenRegion.getOps()) {
    // Note: This prints a superfluous semicolon if the terminating yield op has
    // zero results.
    if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/true)))
      return failure();
  }

  os.unindent() << "}";

  Region &elseRegion = ifOp.getElseRegion();
  if (!elseRegion.empty()) {
    os << " else {\n";
    os.indent();

    for (Operation &op : elseRegion.getOps()) {
      // Note: This prints a superfluous semicolon if the terminating yield op
      // has zero results.
      if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/true)))
        return failure();
    }

    os.unindent() << "}";
  }

  return success();
}

static LogicalResult printOperation(CEmitter &emitter, scf::YieldOp yieldOp) {
  raw_ostream &os = emitter.ostream();
  Operation &parentOp = *yieldOp.getOperation()->getParentOp();

  if (yieldOp.getNumOperands() != parentOp.getNumResults()) {
    return yieldOp.emitError("number of operands does not to match the number "
                             "of the parent op's results");
  }

  if (failed(interleaveWithError(
          llvm::zip(parentOp.getResults(), yieldOp.getOperands()),
          [&](auto pair) -> LogicalResult {
            auto result = std::get<0>(pair);
            auto operand = std::get<1>(pair);
            os << emitter.getOrCreateName(result) << " = ";

            if (!emitter.hasValueInScope(operand))
              return yieldOp.emitError("operand value not in scope");
            os << emitter.getContextMapping(emitter.getOrCreateName(operand));
            // os << emitter.getOrCreateName(operand);
            return success();
          },
          [&]() { os << ";\n"; })))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// Affine dialect
//===----------------------------------------------------------------------===//
static LogicalResult printAffineExpr(CEmitter &emitter, AffineExpr expr,
                                     OperandRange opr, unsigned offset,
                                     unsigned dim = 0) {
  raw_ostream &os = emitter.ostream();

  switch (expr.getKind()) {
  // Unary Op
  case AffineExprKind::SymbolId: {
    // os << "SYMBOL";
    int opPos = expr.cast<AffineSymbolExpr>().getPosition();
    // os << "\n-------------------------------\n";
    // for (auto i : opr) os << i << "\n";
    // os << (opPos + offset + dim);
    // os << "/*" << opr[opPos + offset + dim] << "*/";
    // os << "-------------------------------\n";

    os << emitter.getContextMapping(emitter.getOrCreateName(opr[opPos + offset + dim])).data();
    return success();
  }

  case AffineExprKind::DimId: {
    // os << "DIM";
    int opPos = expr.cast<AffineDimExpr>().getPosition();
    os << emitter.getOrCreateName(opr[opPos + offset]);
    return success();
  }

  case AffineExprKind::Constant:
    os << "(";
    os << expr.cast<AffineConstantExpr>().getValue();
    os << ")";
    return success();

  // Binary Op
  case AffineExprKind::Add: {
    AffineExpr lhs = expr.cast<AffineBinaryOpExpr>().getLHS();
    AffineExpr rhs = expr.cast<AffineBinaryOpExpr>().getRHS();

    if(rhs.getKind() == AffineExprKind::Constant && rhs.cast<AffineConstantExpr>().getValue() < 0) {
      os << "(";
      if (failed(printAffineExpr(emitter, lhs, opr, offset, dim)))
        return failure();
      os << " - ";
      os << abs(rhs.cast<AffineConstantExpr>().getValue());
      os << ")";
    }
    else {
      os << "(";
      if (failed(printAffineExpr(emitter, lhs, opr, offset, dim)))
        return failure();
      os << " + ";
      if (failed(printAffineExpr(emitter, rhs, opr, offset, dim)))
        return failure();
      os << ")";
    }

    return success();
  }

  case AffineExprKind::Mul: {
    AffineExpr lhs = expr.cast<AffineBinaryOpExpr>().getLHS();
    AffineExpr rhs = expr.cast<AffineBinaryOpExpr>().getRHS();

    os << "(";
    if (failed(printAffineExpr(emitter, lhs, opr, offset, dim)))
      return failure();
    os << " * ";
    if (failed(printAffineExpr(emitter, rhs, opr, offset, dim)))
      return failure();
    os << ")";

    return success();
  }

  case AffineExprKind::FloorDiv: {
    AffineExpr lhs = expr.cast<AffineBinaryOpExpr>().getLHS();
    AffineExpr rhs = expr.cast<AffineBinaryOpExpr>().getRHS();

    os << "floor(";
    if (failed(printAffineExpr(emitter, lhs, opr, offset, dim)))
      return failure();
    os << " / ";
    if (failed(printAffineExpr(emitter, rhs, opr, offset, dim)))
      return failure();
    os << ")";

    return success();
  }

  case AffineExprKind::CeilDiv: {
    AffineExpr lhs = expr.cast<AffineBinaryOpExpr>().getLHS();
    AffineExpr rhs = expr.cast<AffineBinaryOpExpr>().getRHS();

    os << "ceil(";
    if (failed(printAffineExpr(emitter, lhs, opr, offset, dim)))
      return failure();
    os << " / ";
    if (failed(printAffineExpr(emitter, rhs, opr, offset, dim)))
      return failure();
    os << ")";

    return success();
  }

  case AffineExprKind::Mod: {
    AffineExpr lhs = expr.cast<AffineBinaryOpExpr>().getLHS();
    AffineExpr rhs = expr.cast<AffineBinaryOpExpr>().getRHS();

    if (failed(printAffineExpr(emitter, lhs, opr, offset, dim)))
      return failure();
    os << " % ";
    if (failed(printAffineExpr(emitter, rhs, opr, offset, dim)))
      return failure();

    return success();
  }
  }

  return failure();
}

static LogicalResult printOperation(CEmitter &emitter,
                                    AffineForOp affineforop) {
	raw_indented_ostream &os = emitter.ostream();

	auto iter = emitter.getOrCreateName(affineforop.getBody()->getArgument(0));
	
	// check prop. expr. and remove them 
	// removeProp()	

	
	os << "for(";
	if (failed(emitter.emitType(affineforop.getLoc(),
								affineforop.getInductionVar().getType())))
		return failure();

	// Lower bound
	os << " " << iter << " = ";
	if (affineforop.hasConstantLowerBound())
		os << affineforop.getConstantLowerBound();
	else {
		AffineMap m = affineforop.getLowerBound().getMap();
		AffineExpr expr = m.getResult(0);
		unsigned dim = m.getNumDims();
		OperandRange opr = affineforop.getLowerBoundOperands();

		if (failed(printAffineExpr(emitter, expr, opr, 0, dim)))
		return failure();
	}

	os << "; ";

	// Upper bound
	os << iter << " < ";
	if (affineforop.hasConstantUpperBound())
		os << affineforop.getConstantUpperBound();
	else {
		AffineMap m = affineforop.getUpperBound().getMap();
		AffineExpr expr = m.getResult(0);
		unsigned dim = m.getNumDims();
		OperandRange opr = affineforop.getUpperBoundOperands();

		// os << "\n------------------------------\n";
		// os << "num dim: " << m.getNumDims() << "\n";
		// os << "num symbol: " << m.getNumSymbols() << "\n";
		// os << "num control op: " << affineforop.getNumControlOperands() << "\n";
		// os << "num iter op: " << affineforop.getNumIterOperands() << "\n";
		// // os << "map is function os dim: " << m.isFunctionOfDim();

		// os << "opr:\n";
		// for (auto it : opr) os << emitter.getOrCreateName(it) << " ";
		// // for (unsigned i = 0; i < opr.size(); i++) {
		// //   os << emitter.getOrCreateName(it) << ":\n";
		// //   os << "Is "
		// // }

		// os << "\n------------------------------\n";

		if (failed(printAffineExpr(emitter, expr, opr, 0, dim)))
		return failure();
	}
	os << "; ";

	os << iter << " += " << affineforop.getStep() << ")"
		<< " {"
		<< "\n";
	os.indent();

	// print region
	// Region &forRegion = affineforop.region();
	auto regionOps = affineforop.getOps();

	for (auto it = regionOps.begin(); std::next(it) != regionOps.end(); ++it) {
		bool trailingSemicolon =
			!isa<scf::IfOp, scf::ForOp, cf::CondBranchOp, AffineForOp>(*it);
		if (failed(emitter.emitOperation(*it, trailingSemicolon)))
		return failure();
	}
	os.unindent();
	os << "}";

	return success();
}

static LogicalResult printOperation(CEmitter &emitter,
                                    AffineIfOp affineifop) {

  raw_indented_ostream &os = emitter.ostream();

  if (!emitter.shouldDeclareVariablesAtTop()) {
    for (OpResult result : affineifop.getResults()) {
      if (failed(emitter.emitVariableDeclaration(result,
                                                 /*trailingSemicolon=*/true)))
        return failure();
    }
  }


  IntegerSet s = affineifop.getIntegerSet();
  ArrayRef<AffineExpr> exprs = s.getConstraints();
  OperandRange opr = affineifop.getOperands();
  unsigned dim = s.getNumDims();

  os << "if (";
  for(int i = 0; i < (int)exprs.size(); i++) {
    if (i != 0) os << " && ";
    if (failed(printAffineExpr(emitter, exprs[i], opr, 0, dim)))
        return failure();

    if (s.isEq(i)) os << " == 0";
    else os << " >= 0";
  }
  os << ") {\n";
  os.indent();

  Region &thenRegion = affineifop.thenRegion();
  for (Operation &op : thenRegion.getOps()) {
    // Note: This prints a superfluous semicolon if the terminating yield op has
    // zero results.
    if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/true)))
      return failure();
  }

  os.unindent() << "}";

  Region &elseRegion = affineifop.elseRegion();
  if (!elseRegion.empty()) {
    os << " else {\n";
    os.indent();

    for (Operation &op : elseRegion.getOps()) {
      // Note: This prints a superfluous semicolon if the terminating yield op
      // has zero results.
      if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/true)))
        return failure();
    }

    os.unindent() << "}";
  }

  return success();
}

static LogicalResult printOperation(CEmitter &emitter, AffineYieldOp affineyieldOp) {
  raw_ostream &os = emitter.ostream();
  Operation &parentOp = *affineyieldOp.getOperation()->getParentOp();

  if (affineyieldOp.getNumOperands() != parentOp.getNumResults()) {
    return affineyieldOp.emitError("number of operands does not to match the number "
                             "of the parent op's results");
  }

  if (failed(interleaveWithError(
          llvm::zip(parentOp.getResults(), affineyieldOp.getOperands()),
          [&](auto pair) -> LogicalResult {
            auto result = std::get<0>(pair);
            auto operand = std::get<1>(pair);
            os << emitter.getOrCreateName(result) << " = ";

            if (!emitter.hasValueInScope(operand))
              return affineyieldOp.emitError("operand value not in scope");
            os << emitter.getContextMapping(emitter.getOrCreateName(operand));
            // os << emitter.getOrCreateName(operand);
            return success();
          },
          [&]() { os << ";\n"; })))
    return failure();

  return success();
}

static LogicalResult printOperation(CEmitter &emitter,
                                    AffineStoreOp affinestoreop) {
  raw_ostream &os = emitter.ostream();

  // Operands
  Operation::operand_range opr = affinestoreop.getOperands();

  // Get number of the dimensions
  AffineMap m = affinestoreop.getAffineMapAttr().getAffineMap();
  unsigned dim = m.getResults().size();

  // Array name
  std::string str(emitter.getOrCreateName(affinestoreop.getOperand(1)));
  // os << emitter.getOrCreateName(affinestoreop.getOperand(1));
  
  // Dimensions
  for (int i = 0; i < (int)dim; i++) {
    AffineExpr expr = m.getResult(i);
    unsigned dim = m.getNumDims();

    str += "[" + emitter.getAffineExprStr(emitter, expr, opr, 2, dim) + "]";
    // os << "[";
    // if (failed(printAffineExpr(emitter, expr, opr, 2, dim)))
    //   return failure();
    // os << "]";
  }
  os << str;
  // os << " = " << emitter.getOrCreateName(affinestoreop.getOperand(0));
  std::string rhs = emitter.getContextMapping(emitter.getOrCreateName(affinestoreop.getOperand(0))).data();
  os << " = " << rhs;
  
//   Operation *operation = affinestoreop.getOperation();

  // affinestoreop.emitWarning() << str << " : " << emitter.getContextMapping(emitter.getOrCreateName(affinestoreop.getOperand(0)));

  // StringRef tmp(str);
  // emitter.createContextMapping(tmp, "(" + std::string(emitter.getContextMapping(emitter.getOrCreateName(affinestoreop.getOperand(0)))) + ")");

  // check lifetime
  std::string r = emitter.getOrCreateName(affinestoreop.getOperand(1)).data();
  
  emitter.deleteKill(r);
  
  return success();
}

static LogicalResult printOperation(CEmitter &emitter, AffineLoadOp affineloadop) {
	raw_ostream &os = emitter.ostream();

	Operation *operation = affineloadop.getOperation();
	if (failed(emitter.emitAssignPrefix(*operation))) return failure();

	// Operands
	Operation::operand_range opr = affineloadop.getOperands();

	// Get number of the dimensions
	AffineMap m = affineloadop.getAffineMapAttr().getAffineMap();
	unsigned dim = m.getResults().size();


	std::string str;
	// Array name
	str += emitter.getOrCreateName(affineloadop.getOperand(0)).str();
	// os << emitter.getOrCreateName(affineloadop.getOperand(0));

	// Dimensions
	for (int i = 0; i < (int)dim; i++) {
		AffineExpr expr = m.getResult(i);
		unsigned dim = m.getNumDims();

		str += "[" + emitter.getAffineExprStr(emitter, expr, opr, 1, dim) + "]";
	}

	os << str;
	str = emitter.getContextMapping(str).data();

	emitter.createContextMapping(emitter.getOrCreateName(operation -> getResult(0)), "(" + str + ")");
	std::set<std::string> s;
	s.insert(emitter.getOrCreateName(affineloadop.getOperand(0)).str());

	// os << "/* updateKill " << emitter.getOrCreateName(operation -> getResult(0)) << " : ";
	// for(auto iter = s.begin(); iter != s.end(); iter++) {
	//   os << " " << *iter;
	// }
	// os << "*/\n";
	emitter.updateKill(emitter.getOrCreateName(operation -> getResult(0)).data(), s);

	std::set<std::string> aho = emitter.getKill(emitter.getOrCreateName(operation -> getResult(0)).data());
	// os << "/* prevent aho[" << emitter.getOrCreateName(operation -> getResult(0)) << "]: ";
	// for(auto iter : aho) os << " " << iter;
	// os << "*/";

  llvm::outs() << "\n------------------------\n";
  llvm::outs() << "Affineloadop operand use info: \n";
  auto expr = affineloadop.getAffineMap();
  const auto& ur = affineloadop.getResult().getUses();
  llvm::outs() << "  " << emitter.getOrCreateName(affineloadop.getResult()) << ":\n";
  for(auto urit = ur.begin(); urit != ur.end(); urit++) {
    llvm::outs() << "    " << urit -> getOwner() -> getName() << " at " << urit ->getOwner() -> getLoc() << "\n";
  }
  {
  const auto& usr = affineloadop.getOperand(0).getUsers();
  llvm::outs() << "  " << emitter.getOrCreateName(affineloadop.getOperand(0)) << ":\n";
  for(auto usrit = usr.begin(); usrit != usr.end(); usrit++) {
    llvm::outs() << "    " << usrit -> getName() << " at " << usrit -> getLoc() << "\n";
  }
  }
  llvm::outs() << "------------------------\n";

	return success();
}

//===----------------------------------------------------------------------===//
// cf dialect
//===----------------------------------------------------------------------===//
static LogicalResult printOperation(CEmitter &emitter, cf::BranchOp branchOp) {
	raw_ostream &os = emitter.ostream();
	Block &successor = *branchOp.getSuccessor();

	for (auto pair :
		llvm::zip(branchOp.getOperands(), successor.getArguments())) {
		Value &operand = std::get<0>(pair);
		BlockArgument &argument = std::get<1>(pair);
		os << emitter.getOrCreateName(argument) << " = "
		<< emitter.getOrCreateName(operand) << ";\n";
	}

	os << "goto ";
	if (!(emitter.hasBlockInScope(successor)))
		return branchOp.emitOpError("unable to find label for successor block");
	os << emitter.getOrCreateName(successor);
	return success();
}

static LogicalResult printOperation(CEmitter &emitter,
                                    cf::CondBranchOp condBranchOp) {
	raw_ostream &os = emitter.ostream();
	Block &trueSuccessor = *condBranchOp.getTrueDest();
	Block &falseSuccessor = *condBranchOp.getFalseDest();

	os << "if (" << emitter.getOrCreateName(condBranchOp.getCondition())
		<< ") {\n";

	// If condition is true.
	for (auto pair : llvm::zip(condBranchOp.getTrueOperands(),
								trueSuccessor.getArguments())) {
		Value &operand = std::get<0>(pair);
		BlockArgument &argument = std::get<1>(pair);
		os << emitter.getOrCreateName(argument) << " = "
		<< emitter.getOrCreateName(operand) << ";\n";
	}

	os << "goto ";
	if (!(emitter.hasBlockInScope(trueSuccessor))) {
		return condBranchOp.emitOpError("unable to find label for successor block");
	}
	os << emitter.getOrCreateName(trueSuccessor) << ";\n";
	os << "} else {\n";
	// If condition is false.
	for (auto pair : llvm::zip(condBranchOp.getFalseOperands(),
								falseSuccessor.getArguments())) {
		Value &operand = std::get<0>(pair);
		BlockArgument &argument = std::get<1>(pair);
		os << emitter.getOrCreateName(argument) << " = "
		<< emitter.getOrCreateName(operand) << ";\n";
	}

	os << "goto ";
	if (!(emitter.hasBlockInScope(falseSuccessor))) {
		return condBranchOp.emitOpError()
			<< "unable to find label for successor block";
	}
	os << emitter.getOrCreateName(falseSuccessor) << ";\n";
	os << "}";

	return success();
}

//===----------------------------------------------------------------------===//
// Builtin Dialect
//===----------------------------------------------------------------------===//
static LogicalResult printOperation(CEmitter &emitter, ModuleOp moduleOp) {
	CEmitter::Scope scope(emitter);
	// moduleOp.emitError("");
	for (Operation &op : moduleOp) {
		if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/false)))
			return failure();
	}
	return success();
}

//===----------------------------------------------------------------------===//
// Sponge Dialect
//===----------------------------------------------------------------------===//
/*
static LogicalResult printOperation(CEmitter &emitter,
                                    miniemitc::CallOp callOp) {
  raw_ostream &os = emitter.ostream();
  Operation &op = *callOp.getOperation();

  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  os << callOp.callee();

  auto emitArgs = [&](Attribute attr) -> LogicalResult {
    if (auto t = attr.dyn_cast<IntegerAttr>()) {
      // Index attributes are treated specially as operand index.
      if (t.getType().isIndex()) {
        int64_t idx = t.getInt();
        if ((idx < 0) || (idx >= op.getNumOperands()))
          return op.emitOpError("invalid operand index");
        if (!emitter.hasValueInScope(op.getOperand(idx)))
          return op.emitOpError("operand ")
                 << idx << "'s value not defined in scope";
        os << emitter.getOrCreateName(op.getOperand(idx));
        return success();
      }
    }
    if (failed(emitter.emitAttribute(op.getLoc(), attr)))
      return failure();

    return success();
  };

  if (callOp.template_args()) {
    os << "<";
    if (failed(interleaveCommaWithError(*callOp.template_args(), os, emitArgs)))
      return failure();
    os << ">";
  }

  os << "(";

  LogicalResult emittedArgs =
      callOp.args() ? interleaveCommaWithError(*callOp.args(), os, emitArgs)
                    : emitter.emitOperands(op);
  if (failed(emittedArgs))
    return failure();
  os << ")";
  return success();
}

static LogicalResult printOperation(CEmitter &emitter,
                                    miniemitc::PragmaOp pragmaop) {
  raw_ostream &os = emitter.ostream();

  os << "#pragma " << pragmaop.pragma();

  return success();
}

static LogicalResult printOperation(CEmitter &emitter,
                                    miniemitc::IncludeOp includeOp) {
  raw_ostream &os = emitter.ostream();

  os << "#include ";
  if (includeOp.is_std_include())
    os << "<" << includeOp.include() << ">";
  else
    os << "\"" << includeOp.include() << "\"";

  return success();
}
*/