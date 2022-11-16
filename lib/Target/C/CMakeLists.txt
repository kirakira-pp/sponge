add_mlir_translation_library(MLIRTargetC
  TranslateRegistration.cpp
  TranslateToC.cpp
  CEmitter.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/sponge/Target/C

  LINK_LIBS PUBLIC
  sponge
  MLIRIR
  MLIRSCF
  MLIRAffine
  MLIRTensor
  MLIRSupport
  MLIRTranslateLib
  )
