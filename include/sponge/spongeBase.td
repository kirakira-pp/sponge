//===- spongeBase.td - sponge dialect base -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SPONGE_BASE
#define SPONGE_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// sponge dialect definition.
//===----------------------------------------------------------------------===//

def sponge_Dialect : Dialect {
    let name = "sponge";
    let summary = "";
    let description = [{}];
    let cppNamespace = "sponge";
}

//===----------------------------------------------------------------------===//
// Base sponge operation definition.
//===----------------------------------------------------------------------===//

class sponge_Op<string mnemonic, list<Trait> traits = []> :
        Op<sponge_Dialect, mnemonic, traits>;

#endif // SPONGE_BASE

