set(LLVM_LINK_COMPONENTS Support)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(sponge-translate sponge-translate.cpp)

llvm_update_compile_flags(sponge-translate)
target_link_libraries(sponge-translate
  PRIVATE
  ${dialect_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTranslateLib
  MLIRSupport
  MLIRTargetC
  )
mlir_check_link_libraries(sponge-translate)
