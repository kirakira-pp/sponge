#include "sponge/Target/C/CEmitter.h"
#include "sponge/Target/C/PrinterDetail.h"

CEmitter::CEmitter(raw_ostream &os, bool declareVariablesAtTop) : 
				   declareVarAtTop(declareVariablesAtTop), os(os) {
	valueInScopeCount.push(0);
	labelInScopeCount.push(0);
}

std::string CEmitter::typeAttrToString(Location loc, Attribute attr) {
	// Integer family
	auto getIntStr = [&](APInt val, bool isUnsigned) -> std::string {
		if(val.getBitWidth() == 1U) {
			if(val.getBoolValue()) return "1";
			else return "0";
		}
		else {
			SmallString<128U> strValue;
			val.toString(strValue, 10, !isUnsigned, true);
		}
		// Any special type of integer will go here
		return "A SPECIAL INTEGER/BOOL WHICH getIntStr CANNOT HANDLE.";
	};
	if(auto iAttr = attr.dyn_cast<IntegerAttr>()) {
		if(auto iType = iAttr.getType().dyn_cast<IntegerType>()) {
			return getIntStr(iAttr.getValue(), iType.getSignedness());
		}
		if(auto idxType = iAttr.getType().dyn_cast<IndexType>()) {
			return getIntStr(iAttr.getValue(), false);
		}
	}

	// Float family
	auto getFloatStr = [&](APFloat val) -> std::string {
		if (val.isFinite()) {
			SmallString<128U> strValue;
			val.toString(strValue, 0U, 18U, true);
			return strValue.str().str();
		} 
		else if (val.isNaN()) {
			// TODO: Translating a NAN. This may not work properly.
			emitError(loc, "Translating a NAN. This may not work properly.");
			return "NAN";
		} 
		else if (val.isInfinity()) {
			// TODO: Translating an INFINITY. This may not work properly.
			emitError(loc, "Translating an INFINITY. This may not work properly.");
			
			if (val.isNegative()) return "-INFINITY";
			return "INFINITY";
		}

		return "A SPECIAL FLOAT THAT IS NOT SUPPORTED.";
	};

	if(auto fAttr = attr.dyn_cast<FloatAttr>()) {
		return getFloatStr(fAttr.getValue());
	}
	// if(auto dAttr = attr.dyn_cast<DenseFPElementsAttr>()) {
	// 	return "AN ARRAY OF FLOATS";
	// }
	// if(auto iAttr = attr.dyn_cast<IntegerAttr>()) {
	// 	if (auto iType = iAttr.getType().dyn_cast<IntegerType>()) {
	// 	return getIntStr(iAttr.getValue(), shouldMapToUnsigned(iType.getSignedness()));
	// 	}
	// 	if (auto iType = iAttr.getType().dyn_cast<IndexType>()) {
	// 	return getIntStr(iAttr.getValue(), false);
	// 	}
	// }
	// if(auto dAttr = attr.dyn_cast<DenseIntElementsAttr>()) {
	// 	return "AN ARRAY OF INTEGERS";
	// } 

	return "AN KNOWN TYPE QQ";
}

LogicalResult CEmitter::emitAttribute(Location loc, Attribute attr) {
	// Integer family
	if(auto iAttr = attr.dyn_cast<IntegerAttr>()) {
		std::string result = typeAttrToString(loc, attr);
		os << result;
		return success();
	}

	// Float family
	if(auto fAttr = attr.dyn_cast<FloatAttr>()) {
		std::string result = typeAttrToString(loc, attr);
		os << result;
		return success();
	}

	// Affine map
	if(auto mpAttr = attr.dyn_cast<AffineMapAttr>()) {
		// TODO: Translating an affine map. This is not supported right now.
		emitWarning(loc, "Translating an affine map. This is not supported right now.");
		
		return success();
	}

	// String 
	if(auto sAttr = attr.dyn_cast<StringAttr>()) {
		// TODO: Translating a string. This is not supported right now.
		emitWarning(loc, "Translating a string. This is not supported right now.");
		
		return success();
	}

	// Symbol reference
	if(auto srAttr = attr.dyn_cast<SymbolRefAttr>()) {
		// TODO: Translating a symbol ref. This is not supported right now.
		emitWarning(loc, "Translating a symbol ref. This is not supported right now.");
		
		return success();
	}

	// TODO: Unknown attributes are encountered.
	emitError(loc, "Unknown attributes are encountered.");


	return success();
}

LogicalResult CEmitter::emitOperation(Operation& op, bool trailingSemicolon) {
	LogicalResult status =
	llvm::TypeSwitch<Operation *, LogicalResult>(&op)
		// buildin op
		.Case<ModuleOp>([&](auto op) { return printOperation(*this, op); })
		// affine ops.
		.Case<AffineForOp, AffineIfOp, AffineYieldOp, AffineStoreOp, AffineLoadOp>(
		    [&](auto op) { return printOperation(*this, op); })
		//   miniEmitC ops.
		//   .Case<miniemitc::PragmaOp, miniemitc::CallOp, miniemitc::IncludeOp>(
		//       [&](auto op) { return printOperation(*this, op); })
		// SCF ops.
		.Case<scf::ForOp, scf::IfOp, scf::YieldOp>(
		    [&](auto op) { return printOperation(*this, op); })
		// func ops.
		.Case<func::CallOp, func::ReturnOp, func::FuncOp>(
			[&](auto op) { return printOperation(*this, op); }) 
		// cf ops.
		.Case<cf::BranchOp, cf::CondBranchOp>(
		    [&](auto op) { return printOperation(*this, op); })
		// math ops. Support scalar only.
		.Case<math::AbsOp, math::SqrtOp, math::ExpOp, math::PowFOp>(
		    [&](auto op) { return printOperation(*this, op); })
		// arith ops.
		.Case<arith::ConstantOp, arith::MulFOp, arith::MulIOp, 
		      arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp, 
		      arith::CmpIOp, arith::CmpFOp, arith::DivFOp, arith::DivSIOp, 
			  arith::RemSIOp, arith::IndexCastOp, arith::SIToFPOp, 
			  arith::UIToFPOp, arith::FPToUIOp, arith::NegFOp, arith::ExtFOp, 
			  arith::TruncIOp, arith::TruncFOp, arith::FPToSIOp, arith::ExtSIOp,
			  arith::SelectOp>(
		    [&](auto op) { return printOperation(*this, op); })
		// memref ops.
		.Case<memref::AllocaOp>(
		    [&](auto op) { if(!shouldDeclareVariablesAtTop()) return printOperation(*this, op); 
		                      trailingSemicolon = false; // nani is this??
		                      return success();})
		.Case<memref::LoadOp, memref::StoreOp>(
		    [&](auto op) { return printOperation(*this, op); })
		.Default([&](Operation *) {
		  return op.emitOpError("unable to find printer for op");
		});

	if (failed(status)) return failure();

	os << (trailingSemicolon ? ";\n" : "\n");
	return success();
}

LogicalResult CEmitter::emitType(Location loc, Type type, StringRef name) {
	// If type is integer family
	if(auto iType = type.dyn_cast<IntegerType>()) {
		switch(iType.getWidth()) {
		case 1:
			os << "bool";
			return success();
		case 16:
			if(iType.isUnsigned()) os << "unsigned ";
			os << "short";
			return success();
		case 32:
			if(iType.isUnsigned()) os << "unsigned ";
			os << "int";
			return success();
		case 64:
			if(iType.isUnsigned()) os << "unsigned ";
			os << "long long";
			return success();

		default:
			emitError(loc, "Not supported interger width.");
		}
	}

	// If the type is float family
	if(auto fType = type.dyn_cast<FloatType>()) {
		switch(fType.getWidth()) {
		case 32:
			os << "float";
			return success();
		case 64:
			os << "double";
			return success();

		default:
			emitError(loc, "Not supported float width.");
		}
	}

	// Index type(add warning)
	if(auto idxType = type.dyn_cast<IndexType>()) {
		// TODO: this should not emit in future.
		emitWarning(loc, "The index type should be ignore by the translator.");
		os << "int";
		return success();
	}

	// Array type family
	if(auto aType = type.dyn_cast<MemRefType>()) {
		if(name.size() == 0) {
			emitError(loc, "The array name should not be empty.");
			return failure();
		}

		// First print name. Then print the dimensions.
		ArrayRef<int64_t> shape = aType.getShape();
		os << name;
		for(auto s : shape) os << "[" << s << "]";

		return success();
	}

	return success();
}

LogicalResult CEmitter::emitTypes(Location loc, llvm::ArrayRef<mlir::Type> types) {
	switch (types.size()) {
	case 0:
		os << "void";
		return success();
	case 1:
		if(failed(emitType(loc, types[0])))
			return failure();
		return success();

	default:
		// TODO: Can this emit a struct?
		emitError(loc, "Cannot emit multiple function return types.\n");
		return failure();
	}
	return success();
}
 
LogicalResult CEmitter::emitLabel(Block& blk) {
	if (!hasBlockInScope(blk))
		return blk.getParentOp()->emitError("label for block not found");

	os << getOrCreateName(blk) << ":\n";
	return success();
}

LogicalResult CEmitter::emitOperands(Operation &op) {
  auto emitOperandName = [&](Value result) -> LogicalResult {
    if (!hasValueInScope(result))
      return op.emitOpError() << "operand value not in scope";
    os << getOrCreateName(result);
    return success();
  };
  return interleaveCommaWithError(op.getOperands(), os, emitOperandName);
}

StringRef CEmitter::getOrCreateName(Value val) {
	if (!valueMapper.count(val))
		valueMapper.insert(val, llvm::formatv("v{0}", ++valueInScopeCount.top()));
	return *valueMapper.begin(val);
}

StringRef CEmitter::getOrCreateName(Block& blk) {
	if (!blockMapper.count(&blk))
		blockMapper.insert(&blk, llvm::formatv("label{0}", ++labelInScopeCount.top()));
	return *blockMapper.begin(&blk);
}

LogicalResult CEmitter::emitAssignPrefix(Operation &op) {
	switch (op.getNumResults()) {
	case 0:
		break;

	case 1: {
		OpResult result = op.getResult(0);
		if (shouldDeclareVariablesAtTop()) {
			if (failed(emitVariableAssignment(result))) return failure();
		} 
		else {
			if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/false)))
				return failure();
			os << " = ";
		}
		break;
	}

	default:
		// TODO: how to translate a correct tuple type in C?
		op.emitWarning("how to translate a correct tuple type in C?\n");
		if (!shouldDeclareVariablesAtTop()) {
			for (OpResult result : op.getResults()) {
				if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/true)))
				return failure();
			}
		}
		os << "std::tie(";
		interleaveComma(op.getResults(), os,
						[&](Value result) { os << getOrCreateName(result); });
		os << ") = ";
	}
	return success();
}

LogicalResult CEmitter::emitVariableAssignment(OpResult result) {
	// If it hasn't been declared, error.
	if (!hasValueInScope(result))
		return result.getDefiningOp()->emitOpError(
			   "result variable for the operation has not been declared");

	os << getOrCreateName(result) << " = ";
	return success();
}

LogicalResult CEmitter::emitVariableDeclaration(OpResult result,
                                                bool trailingSemicolon) {
	// If it has been declared, error.
	if (hasValueInScope(result))
		return result.getDefiningOp()->emitError(
			   "result variable for the operation already declared");

	if (auto aType = result.getType().dyn_cast<MemRefType>()) {
		if (failed(emitType(result.getOwner()->getLoc(), result.getType(),
							getOrCreateName(result))))
			return failure();
	}
	else {
		if (failed(emitType(result.getOwner()->getLoc(), result.getType())))
			return failure();
		os << " " << getOrCreateName(result);
	}

	if (trailingSemicolon) os << ";\n";
	return success();
}

void CEmitter::createContextMapping(StringRef idx, std::string expr) {
	// llvm::outs() << "\nINFO: \nidx: " << idx << " expr: " << expr << "\n";
	if(!contextMapper.count(idx)) {
		// contextMapper.insert(idx, expr);	
		contextMapper.emplace(idx, expr);
	}
}

std::string CEmitter::getContextMapping(StringRef idx) {
	// llvm::outs() << "\nINFO: \nidx: " << idx << "\n";

	// if(contextMapper.count(idx))
		// return *contextMapper.begin(idx.str());
	auto iter = contextMapper.find(idx);
	if(iter != contextMapper.end())
		return iter -> second;

	return idx.str();
}

std::set<std::string> CEmitter::getKill(std::string idx) {
	auto iter = contextMapperSet.find(idx);
	if(iter != contextMapperSet.end()) return iter -> second;

	return {};
}

void CEmitter::updateKill(StringRef idx, std::set<std::string> s) {
	contextMapperSet.emplace(idx, s);
	// contextMapperSet.insert({idx, s});
	return ;
}

void CEmitter::deleteKill(std::string idx) {
	// auto iter = contextMapperSet.find(idx);
	// contextMapperSet.erase(iter);
	std::vector<StringRef> remove_list;

	for(auto cms : contextMapperSet) {
		// llvm::outs() << "/*" << cms.first << ":";
		// for(auto s : cms.second) llvm::outs() << " " << s;
		// llvm::outs() << "*/\n";

		if(cms.second.find(idx) != cms.second.end()) {
			// llvm::outs() << "/*Should be remove*/\n";
			remove_list.push_back(cms.first);
		}
	}

	for(auto rl : remove_list) {
		contextMapperSet.erase(rl);
		contextMapper.erase(rl);
	}

	return ;
}

std::string CEmitter::getAffineExprStr(CEmitter &emitter, AffineExpr expr, 
									   OperandRange opr, unsigned offset, unsigned dim) {
	switch (expr.getKind()) {
	// Unary Op
	case AffineExprKind::SymbolId: {
		int opPos = expr.cast<AffineSymbolExpr>().getPosition();
		std::string str(emitter.getContextMapping(emitter.getOrCreateName(opr[opPos + offset + dim]).str()));
		return str;
	}

	case AffineExprKind::DimId: {
		int opPos = expr.cast<AffineDimExpr>().getPosition();
		std::string str(emitter.getContextMapping(emitter.getOrCreateName(opr[opPos + offset]).str()));
		return str;
	}

	case AffineExprKind::Constant: {
		std::string str(std::to_string(expr.cast<AffineConstantExpr>().getValue()));
		return str;
	}

	// Binary Op
	case AffineExprKind::Add: {
		AffineExpr lhs = expr.cast<AffineBinaryOpExpr>().getLHS();
		AffineExpr rhs = expr.cast<AffineBinaryOpExpr>().getRHS();

		if(rhs.getKind() == AffineExprKind::Constant && 
		   rhs.cast<AffineConstantExpr>().getValue() < 0) {
			int val = abs(rhs.cast<AffineConstantExpr>().getValue());
			std::string str("(" + getAffineExprStr(emitter, lhs, opr, offset, dim) + " - " + std::to_string(val) + ")");

			return str;
		}
		else {
		std::string str( "(" + getAffineExprStr(emitter, lhs, opr, offset, dim) + 
						 " + " + getAffineExprStr(emitter, rhs, opr, offset, dim) + ")");
		return str;
		}

		return "";
	}

	case AffineExprKind::Mul: {
		AffineExpr lhs = expr.cast<AffineBinaryOpExpr>().getLHS();
		AffineExpr rhs = expr.cast<AffineBinaryOpExpr>().getRHS();

		std::string str( "(" + getAffineExprStr(emitter, lhs, opr, offset, dim) + 
						 " * " + getAffineExprStr(emitter, rhs, opr, offset, dim) + ")");
		return str;
	}

	case AffineExprKind::FloorDiv: {
		AffineExpr lhs = expr.cast<AffineBinaryOpExpr>().getLHS();
		AffineExpr rhs = expr.cast<AffineBinaryOpExpr>().getRHS();

		std::string str(         getAffineExprStr(emitter, lhs, opr, offset, dim) + 
						 " / " + getAffineExprStr(emitter, rhs, opr, offset, dim));

		return str;
	}

	case AffineExprKind::CeilDiv: {
		AffineExpr lhs = expr.cast<AffineBinaryOpExpr>().getLHS();
		AffineExpr rhs = expr.cast<AffineBinaryOpExpr>().getRHS();

		std::string str( "ceil(" + getAffineExprStr(emitter, lhs, opr, offset, dim) + 
						  " / " + getAffineExprStr(emitter, rhs, opr, offset, dim) + ")");

		return str;
	}

	case AffineExprKind::Mod: {
		AffineExpr lhs = expr.cast<AffineBinaryOpExpr>().getLHS();
		AffineExpr rhs = expr.cast<AffineBinaryOpExpr>().getRHS();

		std::string str( "(" + getAffineExprStr(emitter, lhs, opr, offset, dim) + 
						 " % " + getAffineExprStr(emitter, rhs, opr, offset, dim) + ")");

		return str;
	}
	}

	return "UNSUPPORT AFFINE EXPR";
}
