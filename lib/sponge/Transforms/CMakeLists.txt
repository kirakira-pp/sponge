add_mlir_library(MLIRspongeTransforms
PassesImplementation.cpp

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

