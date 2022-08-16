get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIROptLib
    MLIRspongeTransforms
    )
add_llvm_executable(sponge-opt sponge-opt.cpp)

# Not sure what is it
install(TARGETS sponge-opt
EXPORT spongeTargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT sponge-opt)

llvm_update_compile_flags(sponge-opt)
target_link_libraries(sponge-opt PRIVATE ${LIBS})
