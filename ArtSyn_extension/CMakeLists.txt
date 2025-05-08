cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(my_extension)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# add_executable(example-app cuda_test.cpp)
# target_link_libraries(example-app "${TORCH_LIBRARIES}")
# set_property(TARGET example-app PROPERTY CXX_STANDARD 17)


# Include directories
include_directories(
    "godot-cpp/include"
    "godot-cpp/bin"
    "godot-cpp/gen/include"
    "godot-cpp/gdextension"
)

# Find libraries
link_directories(
    "godot-cpp/bin"
)

# Create shared library
add_library(my_extension SHARED my_custom_node.cpp)

# Set output name for the library
set_target_properties(my_extension PROPERTIES 
    OUTPUT_NAME "my_extension"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    CXX_STANDARD 17
)

# Link libraries
target_link_libraries(my_extension
    -l:libgodot-cpp.linux.template_debug.x86_64.a
    "${TORCH_LIBRARIES}"
    # torch
    # torch_cpu
    # torch_cuda
    # c10
    # c10_cuda
    # cudart
    # cublas
    # cudnn  # Commented out as in the original
)


# Add linker flag to prevent undefined symbols
if(UNIX AND NOT APPLE)
    target_link_options(my_extension PRIVATE "-Wl,--no-undefined")
endif()
