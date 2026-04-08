cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED INPUT_FILE)
    message(FATAL_ERROR "INPUT_FILE is required")
endif()

if(NOT DEFINED OUTPUT_DIR)
    message(FATAL_ERROR "OUTPUT_DIR is required")
endif()

if(NOT EXISTS "${INPUT_FILE}")
    message(FATAL_ERROR "INPUT_FILE does not exist: ${INPUT_FILE}")
endif()

file(MAKE_DIRECTORY "${OUTPUT_DIR}")

find_program(PATCHELF_EXECUTABLE patchelf)

function(copy_runtime_library_with_soname runtime_library)
    if(NOT EXISTS "${runtime_library}")
        return()
    endif()

    file(COPY "${runtime_library}" DESTINATION "${OUTPUT_DIR}" FOLLOW_SYMLINK_CHAIN)

    execute_process(
        COMMAND readelf -d "${runtime_library}"
        RESULT_VARIABLE _readelf_result
        OUTPUT_VARIABLE _readelf_output
        ERROR_QUIET
    )

    if(NOT _readelf_result EQUAL 0)
        return()
    endif()

    string(REGEX MATCH "SONAME[^[]*\\[([^]]+)\\]" _soname_match "${_readelf_output}")
    if(NOT CMAKE_MATCH_1)
        return()
    endif()

    get_filename_component(_runtime_library_name "${runtime_library}" NAME)
    if(_runtime_library_name STREQUAL "${CMAKE_MATCH_1}")
        return()
    endif()

    set(_soname_link "${OUTPUT_DIR}/${CMAKE_MATCH_1}")
    if(NOT EXISTS "${_soname_link}")
        file(CREATE_LINK "${_runtime_library_name}" "${_soname_link}" SYMBOLIC)
    endif()
endfunction()

function(normalize_onnxruntime_rpath copied_runtime_library)
    if(NOT PATCHELF_EXECUTABLE OR NOT EXISTS "${copied_runtime_library}" OR IS_SYMLINK "${copied_runtime_library}")
        return()
    endif()

    get_filename_component(_copied_runtime_library_name "${copied_runtime_library}" NAME)
    if(NOT _copied_runtime_library_name MATCHES "^libonnxruntime.*\\.so(\\..*)?$")
        return()
    endif()

    execute_process(
        COMMAND "${PATCHELF_EXECUTABLE}" --set-rpath "$ORIGIN" "${copied_runtime_library}"
        RESULT_VARIABLE _patchelf_result
        ERROR_VARIABLE _patchelf_error
    )

    if(NOT _patchelf_result EQUAL 0)
        string(STRIP "${_patchelf_error}" _patchelf_error)
        message(WARNING "Failed to normalize RPATH for ${copied_runtime_library}: ${_patchelf_error}")
    endif()
endfunction()

get_filename_component(_input_name "${INPUT_FILE}" NAME)
if(NOT DEFINED ARTSYN_EXTENSION_STEM OR ARTSYN_EXTENSION_STEM STREQUAL "")
    set(ARTSYN_EXTENSION_STEM "my_extension")
endif()

file(GLOB _existing_runtime_libs "${OUTPUT_DIR}/*.so*")
foreach(_existing_runtime_lib IN LISTS _existing_runtime_libs)
    get_filename_component(_existing_name "${_existing_runtime_lib}" NAME)
    if(_existing_name STREQUAL "${_input_name}")
        continue()
    endif()

    if(_existing_name MATCHES "^lib${ARTSYN_EXTENSION_STEM}(\\..+)?\\.so$")
        continue()
    endif()

    if(_existing_name MATCHES "^${ARTSYN_EXTENSION_STEM}(\\..+)?\\.dll$")
        continue()
    endif()

    if(_existing_name MATCHES "^lib${ARTSYN_EXTENSION_STEM}(\\..+)?\\.dylib$")
        continue()
    endif()

    if(_existing_name MATCHES "^${ARTSYN_EXTENSION_STEM}(\\..+)?\\.so$")
        continue()
    endif()

    if(_existing_name MATCHES "\\.so" OR _existing_name MATCHES "\\.dll" OR _existing_name MATCHES "\\.dylib")
        file(REMOVE "${_existing_runtime_lib}")
    endif()
endforeach()

set(_dependency_search_dirs)
foreach(_candidate_dir
    "${ARTSYN_TORCH_ROOT}/lib"
    "${ARTSYN_ONNX_RUNTIME_LIBRARY_DIR}"
    "${CUDAToolkit_LIBRARY_DIR}"
    "${CUDAToolkit_LIBRARY_ROOT}"
    "${CUDAToolkit_LIBRARY_ROOT}/lib64"
    "${CUDAToolkit_TARGET_DIR}/lib"
    "${CUDAToolkit_TARGET_DIR}/lib64"
)
    if(_candidate_dir AND EXISTS "${_candidate_dir}")
        list(APPEND _dependency_search_dirs "${_candidate_dir}")
    endif()
endforeach()

foreach(_candidate_dir IN LISTS ARTSYN_EXTRA_RUNTIME_LIBRARY_DIRS)
    if(_candidate_dir AND EXISTS "${_candidate_dir}")
        list(APPEND _dependency_search_dirs "${_candidate_dir}")
    endif()
endforeach()
list(REMOVE_DUPLICATES _dependency_search_dirs)

set(_direct_runtime_libraries)
if(ARTSYN_ONNX_RUNTIME_LIBRARY AND EXISTS "${ARTSYN_ONNX_RUNTIME_LIBRARY}")
    list(APPEND _direct_runtime_libraries "${ARTSYN_ONNX_RUNTIME_LIBRARY}")
endif()
if(ARTSYN_ONNX_RUNTIME_LIBRARY_DIR AND EXISTS "${ARTSYN_ONNX_RUNTIME_LIBRARY_DIR}")
    file(GLOB _onnx_provider_libraries
        "${ARTSYN_ONNX_RUNTIME_LIBRARY_DIR}/libonnxruntime_providers*.so*"
    )
    list(APPEND _direct_runtime_libraries ${_onnx_provider_libraries})
endif()
list(REMOVE_DUPLICATES _direct_runtime_libraries)

set(_bundled_dependencies)
foreach(_direct_runtime_library IN LISTS _direct_runtime_libraries)
    if(EXISTS "${_direct_runtime_library}")
        copy_runtime_library_with_soname("${_direct_runtime_library}")
        list(APPEND _bundled_dependencies "${_direct_runtime_library}")
    endif()
endforeach()

file(GET_RUNTIME_DEPENDENCIES
    RESOLVED_DEPENDENCIES_VAR _resolved_dependencies
    UNRESOLVED_DEPENDENCIES_VAR _unresolved_dependencies
    LIBRARIES "${INPUT_FILE}" ${_direct_runtime_libraries}
    DIRECTORIES ${_dependency_search_dirs}
    PRE_EXCLUDE_REGEXES
        "api-ms-"
        "ext-ms-"
    POST_EXCLUDE_REGEXES
        ".*/ld-linux[^/]*\\.so.*"
        ".*/libc\\.so.*"
        ".*/libdl\\.so.*"
        ".*/libgcc_s\\.so.*"
        ".*/libm\\.so.*"
        ".*/libpthread\\.so.*"
        ".*/librt\\.so.*"
        ".*/libstdc\\+\\+\\.so.*"
)

set(_bundled_dependency_regexes
    ".*/libaoti_custom_ops\\.so.*"
    ".*/libbackend_with_compiler\\.so.*"
    ".*/libc10.*\\.so.*"
    ".*/libcaffe2.*\\.so.*"
    ".*/libcublas.*\\.so.*"
    ".*/libcurand.*\\.so.*"
    ".*/libcudart.*\\.so.*"
    ".*/libcudnn.*\\.so.*"
    ".*/libcufft.*\\.so.*"
    ".*/libcufile.*\\.so.*"
    ".*/libcusparseLt.*\\.so.*"
    ".*/libgomp.*\\.so.*"
    ".*/libjitbackend_test\\.so.*"
    ".*/libnnapi_backend\\.so.*"
    ".*/libnvrtc.*\\.so.*"
    ".*/libnvToolsExt.*\\.so.*"
    ".*/libonnxruntime.*\\.so.*"
    ".*/libshm\\.so.*"
    ".*/libtorch.*\\.so.*"
)

foreach(_resolved_dependency IN LISTS _resolved_dependencies)
    set(_should_bundle FALSE)
    foreach(_bundled_dependency_regex IN LISTS _bundled_dependency_regexes)
        if(_resolved_dependency MATCHES "${_bundled_dependency_regex}")
            set(_should_bundle TRUE)
            break()
        endif()
    endforeach()

    if(_should_bundle)
        file(COPY "${_resolved_dependency}" DESTINATION "${OUTPUT_DIR}" FOLLOW_SYMLINK_CHAIN)
        list(APPEND _bundled_dependencies "${_resolved_dependency}")
    endif()
endforeach()

file(GLOB _copied_onnxruntime_libraries "${OUTPUT_DIR}/libonnxruntime*.so*")
foreach(_copied_onnxruntime_library IN LISTS _copied_onnxruntime_libraries)
    normalize_onnxruntime_rpath("${_copied_onnxruntime_library}")
endforeach()

list(LENGTH _bundled_dependencies _bundled_count)
message(STATUS "Bundled ${_bundled_count} runtime libraries into ${OUTPUT_DIR}")

if(_unresolved_dependencies)
    message(WARNING "Unresolved runtime dependencies: ${_unresolved_dependencies}")
endif()
