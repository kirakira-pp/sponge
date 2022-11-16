#include "sponge/Target/C/CEmitter.h"

LogicalResult sponge::translateToC(Operation *op, raw_ostream &os,
                                      bool declareVariablesAtTop) {
  CEmitter emitter(os, declareVariablesAtTop);
  return emitter.emitOperation(*op, /*trailingSemicolon=*/false);
    // return success();
}
