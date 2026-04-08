cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ArtSyn_extension LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(GODOT_CPP_DIR "" CACHE PATH "Path to a godot-cpp checkout")
set(GODOT_CPP_LIBRARY "" CACHE FILEPATH "Path to the prebuilt godot-cpp static library")
set(ARTSYN_TORCH_ROOT "" CACHE PATH "Path to a LibTorch installation root")
set(ARTSYN_ONNX_RUNTIME_ROOT "" CACHE PATH "Path to an ONNX Runtime installation root")
set(ARTSYN_ONNX_RUNTIME_INCLUDE_DIR "" CACHE PATH "Path to the ONNX Runtime headers")
set(ARTSYN_ONNX_RUNTIME_LIBRARY "" CACHE FILEPATH "Path to the ONNX Runtime shared library")
set(ARTSYN_EXTRA_RUNTIME_LIBRARY_DIRS "" CACHE STRING "Semicolon-separated list of extra runtime library directories to search when bundling dependencies")
set(ARTSYN_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../2d-platformer-demo/addons/artsyn/bin" CACHE PATH "Directory for the compiled extension library")
set(ARTSYN_EXTENSION_STEM "my_extension" CACHE STRING "Base file name for the compiled extension without the platform prefix/suffix")
option(ARTSYN_ENABLE_LIBTORCH_BACKEND "Build the LibTorch backend" ON)
option(ARTSYN_BUNDLE_TORCH_LIBS "Copy LibTorch runtime libraries next to the built extension" ON)
option(ARTSYN_BUNDLE_CUDA_RUNTIME "Copy the CUDA runtime library next to the built extension when available" ON)
option(ARTSYN_ENABLE_ONNX_RUNTIME "Build the ONNX backend against ONNX Runtime when the SDK is provided" OFF)
option(ARTSYN_BUNDLE_ONNX_RUNTIME "Copy ONNX Runtime libraries next to the built extension when available" ON)

if(DEFINED ENV{ARTSYN_GODOT_CPP_DIR} AND (NOT GODOT_CPP_DIR OR NOT EXISTS "${GODOT_CPP_DIR}/include"))
    set(GODOT_CPP_DIR "$ENV{ARTSYN_GODOT_CPP_DIR}" CACHE PATH "Path to a godot-cpp checkout" FORCE)
endif()

if(DEFINED ENV{ARTSYN_TORCH_ROOT} AND (NOT ARTSYN_TORCH_ROOT OR NOT EXISTS "${ARTSYN_TORCH_ROOT}"))
    set(ARTSYN_TORCH_ROOT "$ENV{ARTSYN_TORCH_ROOT}" CACHE PATH "Path to a LibTorch installation root" FORCE)
endif()

if(DEFINED ENV{ARTSYN_ONNX_RUNTIME_ROOT} AND (NOT ARTSYN_ONNX_RUNTIME_ROOT OR NOT EXISTS "${ARTSYN_ONNX_RUNTIME_ROOT}"))
    set(ARTSYN_ONNX_RUNTIME_ROOT "$ENV{ARTSYN_ONNX_RUNTIME_ROOT}" CACHE PATH "Path to an ONNX Runtime installation root" FORCE)
endif()

if(DEFINED ENV{ARTSYN_ONNX_RUNTIME_INCLUDE_DIR} AND (NOT ARTSYN_ONNX_RUNTIME_INCLUDE_DIR OR NOT EXISTS "${ARTSYN_ONNX_RUNTIME_INCLUDE_DIR}"))
    set(ARTSYN_ONNX_RUNTIME_INCLUDE_DIR "$ENV{ARTSYN_ONNX_RUNTIME_INCLUDE_DIR}" CACHE PATH "Path to the ONNX Runtime headers" FORCE)
endif()

if(DEFINED ENV{ARTSYN_ONNX_RUNTIME_LIBRARY} AND (NOT ARTSYN_ONNX_RUNTIME_LIBRARY OR NOT EXISTS "${ARTSYN_ONNX_RUNTIME_LIBRARY}"))
    set(ARTSYN_ONNX_RUNTIME_LIBRARY "$ENV{ARTSYN_ONNX_RUNTIME_LIBRARY}" CACHE FILEPATH "Path to the ONNX Runtime shared library" FORCE)
endif()

set(_godot_cpp_candidates
    "${GODOT_CPP_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/godot-cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../godot-cpp"
)

set(_resolved_godot_cpp_dir "")
foreach(_candidate IN LISTS _godot_cpp_candidates)
    if(_candidate AND EXISTS "${_candidate}/include" AND EXISTS "${_candidate}/gdextension")
        set(_resolved_godot_cpp_dir "${_candidate}")
        break()
    endif()
endforeach()

if(NOT _resolved_godot_cpp_dir)
    message(FATAL_ERROR
        "Could not find godot-cpp.\n"
        "Set GODOT_CPP_DIR in CMake or ARTSYN_GODOT_CPP_DIR in the environment."
    )
endif()

set(GODOT_CPP_DIR "${_resolved_godot_cpp_dir}" CACHE PATH "Path to a godot-cpp checkout" FORCE)

if(DEFINED ENV{ARTSYN_GODOT_CPP_LIBRARY} AND (NOT GODOT_CPP_LIBRARY OR NOT EXISTS "${GODOT_CPP_LIBRARY}"))
    set(GODOT_CPP_LIBRARY "$ENV{ARTSYN_GODOT_CPP_LIBRARY}" CACHE FILEPATH "Path to the prebuilt godot-cpp static library" FORCE)
endif()

if(NOT GODOT_CPP_LIBRARY)
    set(GODOT_CPP_LIBRARY "${GODOT_CPP_DIR}/bin/libgodot-cpp.linux.template_debug.x86_64.a")
endif()

if(NOT EXISTS "${GODOT_CPP_LIBRARY}")
    message(FATAL_ERROR
        "Could not find the godot-cpp static library.\n"
        "Expected: ${GODOT_CPP_LIBRARY}\n"
        "Set GODOT_CPP_LIBRARY in CMake or ARTSYN_GODOT_CPP_LIBRARY in the environment."
    )
endif()

set(_artsyn_with_libtorch_backend FALSE)
if(ARTSYN_ENABLE_LIBTORCH_BACKEND)
    if(ARTSYN_TORCH_ROOT)
        list(PREPEND CMAKE_PREFIX_PATH
            "${ARTSYN_TORCH_ROOT}"
            "${ARTSYN_TORCH_ROOT}/share/cmake/Torch"
        )
    endif()

    if(DEFINED ENV{ARTSYN_TORCH_DIR} AND (NOT Torch_DIR OR NOT EXISTS "${Torch_DIR}/TorchConfig.cmake"))
        set(Torch_DIR "$ENV{ARTSYN_TORCH_DIR}" CACHE PATH "Path to TorchConfig.cmake" FORCE)
    endif()

    find_package(Torch REQUIRED)
    find_package(CUDAToolkit QUIET)
    separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    set(_artsyn_with_libtorch_backend TRUE)
endif()

if(ARTSYN_ONNX_RUNTIME_ROOT)
    if(NOT ARTSYN_ONNX_RUNTIME_INCLUDE_DIR AND EXISTS "${ARTSYN_ONNX_RUNTIME_ROOT}/include")
        set(ARTSYN_ONNX_RUNTIME_INCLUDE_DIR "${ARTSYN_ONNX_RUNTIME_ROOT}/include" CACHE PATH "Path to the ONNX Runtime headers" FORCE)
    endif()

    if(NOT ARTSYN_ONNX_RUNTIME_LIBRARY)
        find_library(_artsyn_onnxruntime_library
            NAMES onnxruntime
            HINTS
                "${ARTSYN_ONNX_RUNTIME_ROOT}/lib"
                "${ARTSYN_ONNX_RUNTIME_ROOT}/lib64"
            NO_DEFAULT_PATH
        )

        if(_artsyn_onnxruntime_library)
            set(ARTSYN_ONNX_RUNTIME_LIBRARY "${_artsyn_onnxruntime_library}" CACHE FILEPATH "Path to the ONNX Runtime shared library" FORCE)
        endif()
    endif()
endif()

set(_artsyn_with_onnx_runtime FALSE)
if(ARTSYN_ENABLE_ONNX_RUNTIME)
    if(NOT ARTSYN_ONNX_RUNTIME_INCLUDE_DIR OR NOT EXISTS "${ARTSYN_ONNX_RUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
        message(FATAL_ERROR
            "ARTSYN_ENABLE_ONNX_RUNTIME is ON, but ONNX Runtime headers were not found.\n"
            "Set ARTSYN_ONNX_RUNTIME_INCLUDE_DIR or ARTSYN_ONNX_RUNTIME_ROOT."
        )
    endif()

    if(NOT ARTSYN_ONNX_RUNTIME_LIBRARY OR NOT EXISTS "${ARTSYN_ONNX_RUNTIME_LIBRARY}")
        message(FATAL_ERROR
            "ARTSYN_ENABLE_ONNX_RUNTIME is ON, but the ONNX Runtime library was not found.\n"
            "Set ARTSYN_ONNX_RUNTIME_LIBRARY or ARTSYN_ONNX_RUNTIME_ROOT."
        )
    endif()

    set(_artsyn_with_onnx_runtime TRUE)
endif()

set(_artsyn_extension_sources
    my_custom_node.cpp
    onnx_style_transfer_backend.cpp
    style_transfer_backend_factory.cpp
)

if(_artsyn_with_libtorch_backend)
    list(APPEND _artsyn_extension_sources libtorch_style_transfer_backend.cpp)
endif()

add_library(my_extension SHARED ${_artsyn_extension_sources})

target_include_directories(my_extension PRIVATE
    "${GODOT_CPP_DIR}/include"
    "${GODOT_CPP_DIR}/gen/include"
    "${GODOT_CPP_DIR}/gdextension"
)

target_link_libraries(my_extension PRIVATE "${GODOT_CPP_LIBRARY}")

if(_artsyn_with_libtorch_backend)
    target_compile_definitions(my_extension PRIVATE ARTSYN_WITH_LIBTORCH_BACKEND=1)
    target_compile_options(my_extension PRIVATE ${TORCH_CXX_FLAGS_LIST})
    target_link_libraries(my_extension PRIVATE "${TORCH_LIBRARIES}")
endif()

if(_artsyn_with_onnx_runtime)
    target_compile_definitions(my_extension PRIVATE ARTSYN_WITH_ONNX_RUNTIME=1)
    target_include_directories(my_extension PRIVATE "${ARTSYN_ONNX_RUNTIME_INCLUDE_DIR}")
    target_link_libraries(my_extension PRIVATE "${ARTSYN_ONNX_RUNTIME_LIBRARY}")
endif()

set_target_properties(my_extension PROPERTIES
    OUTPUT_NAME "${ARTSYN_EXTENSION_STEM}"
    LIBRARY_OUTPUT_DIRECTORY "${ARTSYN_OUTPUT_DIR}"
    DEBUG_POSTFIX ".debug"
    RELEASE_POSTFIX ".release"
    RELWITHDEBINFO_POSTFIX ".release"
    MINSIZEREL_POSTFIX ".release"
)

if(UNIX AND NOT APPLE)
    set_target_properties(my_extension PROPERTIES
        BUILD_WITH_INSTALL_RPATH ON
        BUILD_RPATH_USE_ORIGIN ON
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
        INSTALL_RPATH_USE_LINK_PATH FALSE
    )
    target_link_options(my_extension PRIVATE "-Wl,--no-undefined")
endif()

if((_artsyn_with_libtorch_backend AND (ARTSYN_BUNDLE_TORCH_LIBS OR ARTSYN_BUNDLE_CUDA_RUNTIME)) OR (ARTSYN_BUNDLE_ONNX_RUNTIME AND _artsyn_with_onnx_runtime))
    if(_artsyn_with_onnx_runtime)
        get_filename_component(_artsyn_onnx_runtime_library_dir "${ARTSYN_ONNX_RUNTIME_LIBRARY}" DIRECTORY)
    else()
        set(_artsyn_onnx_runtime_library_dir "")
    endif()

    add_custom_command(TARGET my_extension POST_BUILD
        COMMAND "${CMAKE_COMMAND}"
            -DINPUT_FILE=$<TARGET_FILE:my_extension>
            -DOUTPUT_DIR=$<TARGET_FILE_DIR:my_extension>
            -DARTSYN_EXTENSION_STEM=${ARTSYN_EXTENSION_STEM}
            -DARTSYN_TORCH_ROOT=${ARTSYN_TORCH_ROOT}
            -DARTSYN_ONNX_RUNTIME_LIBRARY=${ARTSYN_ONNX_RUNTIME_LIBRARY}
            -DARTSYN_ONNX_RUNTIME_LIBRARY_DIR=${_artsyn_onnx_runtime_library_dir}
            -DARTSYN_EXTRA_RUNTIME_LIBRARY_DIRS=${ARTSYN_EXTRA_RUNTIME_LIBRARY_DIRS}
            -DCUDAToolkit_LIBRARY_DIR=${CUDAToolkit_LIBRARY_DIR}
            -DCUDAToolkit_LIBRARY_ROOT=${CUDAToolkit_LIBRARY_ROOT}
            -DCUDAToolkit_TARGET_DIR=${CUDAToolkit_TARGET_DIR}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/copy_runtime_deps.cmake"
        VERBATIM
    )
endif()

message(STATUS "Using godot-cpp from: ${GODOT_CPP_DIR}")
message(STATUS "Using godot-cpp library: ${GODOT_CPP_LIBRARY}")
message(STATUS "Extension output directory: ${ARTSYN_OUTPUT_DIR}")
if(_artsyn_with_libtorch_backend)
    message(STATUS "LibTorch backend build is enabled.")
else()
    message(STATUS "LibTorch backend build is disabled.")
endif()
if(_artsyn_with_onnx_runtime)
    message(STATUS "Using ONNX Runtime from: ${ARTSYN_ONNX_RUNTIME_LIBRARY}")
else()
    message(STATUS "ONNX Runtime backend build is disabled; the 'onnx' backend remains a stub.")
endif()
