# Add a new dialect named sponge. The first parameter 
# is dialect name in mlir. The second parameter is
# namespace in C++.
add_mlir_dialect(sponge sponge)


# Add subdirectory
add_subdirectory(Transforms)
