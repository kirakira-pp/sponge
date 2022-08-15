//===- sponge.h - sponge dialect -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SPONGE_SPONGEDIALECT_H
#define SPONGE_SPONGEDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/StringExtras.h"

// Dialect decls
#include "sponge/spongeDialect.h.inc"

#define GET_OP_CLASSES
#include "sponge/sponge.h.inc"
#define GET_TYPEDEF_CLASSES
#include "sponge/spongeTypes.h.inc"

#endif // SPONGE_SPONGEDIALECT_H

