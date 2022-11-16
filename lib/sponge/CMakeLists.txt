add_mlir_dialect_library(sponge
    sponge.cpp

    # ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/sponge

    DEPENDS
    MLIRspongeIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSideEffectInterfaces
)

add_subdirectory(Transforms)

