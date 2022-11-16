//===- sponge.cpp - sponge dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sponge/sponge.h"

using namespace mlir;
using namespace sponge;

#include "sponge/spongeDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// sponge dialect.
//===----------------------------------------------------------------------===//

void spongeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "sponge/sponge.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "sponge/spongeTypes.cpp.inc"
      >();
}

