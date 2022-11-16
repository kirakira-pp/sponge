add_mlir_library(MLIRspongeTransformsUtils
aff.cpp


  DEPENDS
  #MLIRsponge
  MLIRspongePassesIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
