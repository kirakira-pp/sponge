
//===- CEmitter.h - Helpers to create C++ emitter -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines helpers to emit C++ code using the EmitC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_C_CEMITTER_H
#define MLIR_TARGET_C_CEMITTER_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

// LLVM tools
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/FormatVariadic.h"
// #include "llvm/ADT/iterator_range.h"

// MLIR tools
#include "mlir/Support/IndentedOstream.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IntegerSet.h"
// #include "mlir/IR/Value.h"

// Dialects
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

// std
#include <stack>
#include <map>
#include <set>
#include <vector>

using namespace mlir;

/// Convenience functions to produce interleaved output with functions returning
/// a LogicalResult. This is different than those in STLExtras as functions used
/// on each element doesn't return a string.
template <typename ForwardIterator, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult
interleaveWithError(ForwardIterator begin, ForwardIterator end,
					UnaryFunctor eachFn, NullaryFunctor betweenFn) {
	if (begin == end)
		return success();
	if (failed(eachFn(*begin)))
		return failure();
	++begin;
	for (; begin != end; ++begin) {
		betweenFn();
		if (failed(eachFn(*begin)))
			return failure();
	}
	return success();
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult interleaveWithError(const Container &c,
										 UnaryFunctor eachFn,
										 NullaryFunctor betweenFn) {
	return interleaveWithError(c.begin(), c.end(), eachFn, betweenFn);
}

template <typename Container, typename UnaryFunctor>
inline LogicalResult interleaveCommaWithError(const Container &c, 
											  raw_ostream &os,
											  UnaryFunctor eachFn) {
	return interleaveWithError(c.begin(), c.end(), eachFn, [&]() { os << ", "; });
}

class CEmitter {
public:
	explicit CEmitter(raw_ostream &os, bool declareVariablesAtTop);

	/// Basic translate funtions
	LogicalResult emitOperation(Operation& op, bool trailingSemicolon);
	LogicalResult emitAttribute(Location loc, Attribute attr);
	LogicalResult emitType(Location loc, Type type, StringRef name = "");
	// TODO: Can this emit a struct?
	LogicalResult emitTypes(Location loc, llvm::ArrayRef<mlir::Type> types);
	LogicalResult emitLabel(Block& blk);
	LogicalResult emitAssignPrefix(Operation& op);
	LogicalResult emitVariableAssignment(OpResult result);
	LogicalResult emitVariableDeclaration(OpResult result, bool trailingSemicolon);
	LogicalResult emitOperands(Operation& op);
	std::string typeAttrToString(Location loc, Attribute attr);
	std::string getAffineExprStr(CEmitter &emitter, AffineExpr expr, OperandRange opr, unsigned offset, unsigned dim = 0);


	/// scope maintainance
	StringRef getOrCreateName(Value val);
	StringRef getOrCreateName(Block& blk);
	void createContextMapping(StringRef idx, std::string expr);
	std::string getContextMapping(StringRef idx);
	bool hasValueInScope(Value val) { return valueMapper.count(val); }
	bool hasBlockInScope(Block& blk) { return blockMapper.count(&blk); }

	/// Get class member vars
	bool shouldDeclareVariablesAtTop() { return declareVarAtTop; };
	raw_indented_ostream& ostream() { return os; };

	/// Propagation
	std::set<std::string> getKill(std::string key);
	void updateKill(StringRef key, std::set<std::string> s);
	void deleteKill(std::string idx);

	/// Emit types
	/// print comma

	struct Scope {
	public:
		Scope(CEmitter &emitter) : valueMapperScope(emitter.valueMapper),
								   blockMapperScope(emitter.blockMapper), 
								   contextMapperScope(emitter.contextMapper),
								   contextSetMapperScope(emitter.contextMapperSet),
								   emitter(emitter) {
			emitter.valueInScopeCount.push(emitter.valueInScopeCount.top());
			emitter.labelInScopeCount.push(emitter.labelInScopeCount.top());
		}
		~Scope() {
			emitter.valueInScopeCount.pop();
			emitter.labelInScopeCount.pop();
		}

	private:
		llvm::ScopedHashTableScope<Value, std::string> valueMapperScope;
		llvm::ScopedHashTableScope<Block*, std::string> blockMapperScope;
		// llvm::ScopedHashTableScope<StringRef, std::string> contextMapperScope;
		std::map<StringRef, std::string> contextMapperScope;
		std::map<StringRef, std::set<std::string>> contextSetMapperScope;
		CEmitter &emitter;
};

private:
	bool declareVarAtTop = false;
	raw_indented_ostream os;


	/// Scope maintainance

	// Scope table
	using ValueMapper = llvm::ScopedHashTable<Value, std::string>;
	using BlockMapper = llvm::ScopedHashTable<Block*, std::string>;

	ValueMapper valueMapper;
	BlockMapper blockMapper;

	/// The number of values in the current scope. This is used to declare the
	/// names of values in a scope.
	std::stack<int64_t> valueInScopeCount;
	std::stack<int64_t> labelInScopeCount;

	/// Propagations
	// using ContextMapper = llvm::ScopedHashTable<StringRef, std::string>;
	using ContextMapper    = std::map<StringRef, std::string>;
	using ContextSetMapper = std::map<StringRef, std::set<std::string>>;

	ContextMapper contextMapper;
	ContextSetMapper contextMapperSet;

};

namespace sponge {

/// Translates the given operation to C code. The operation or operations in
/// the region of 'op' need almost all be in EmitC dialect. The parameter
/// 'declareVariablesAtTop' enforces that all variables for op results and block
/// arguments are declared at the beginning of the function.
LogicalResult translateToC(Operation *op, raw_ostream &os,
						   bool declareVariablesAtTop = false);

} // namespace sponge

#endif // MLIR_TARGET_CPP_CPPEMITTER_H
