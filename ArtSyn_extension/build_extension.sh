#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ARTSYN_BUILD_DIR:-$ROOT_DIR/build/linux-debug}"
BUILD_TYPE="${ARTSYN_BUILD_TYPE:-Debug}"
OUTPUT_DIR="${ARTSYN_OUTPUT_DIR:-}"
EXTENSION_STEM="${ARTSYN_EXTENSION_STEM:-}"
ENABLE_LIBTORCH_BACKEND="${ARTSYN_ENABLE_LIBTORCH_BACKEND:-ON}"
BUNDLE_TORCH_LIBS="${ARTSYN_BUNDLE_TORCH_LIBS:-}"
BUNDLE_CUDA_RUNTIME="${ARTSYN_BUNDLE_CUDA_RUNTIME:-}"
ENABLE_ONNX_RUNTIME="${ARTSYN_ENABLE_ONNX_RUNTIME:-}"
BUNDLE_ONNX_RUNTIME="${ARTSYN_BUNDLE_ONNX_RUNTIME:-}"
ONNX_RUNTIME_ROOT="${ARTSYN_ONNX_RUNTIME_ROOT:-}"
ONNX_RUNTIME_INCLUDE_DIR="${ARTSYN_ONNX_RUNTIME_INCLUDE_DIR:-}"
ONNX_RUNTIME_LIBRARY="${ARTSYN_ONNX_RUNTIME_LIBRARY:-}"
EXTRA_RUNTIME_LIBRARY_DIRS="${ARTSYN_EXTRA_RUNTIME_LIBRARY_DIRS:-}"

: "${ARTSYN_GODOT_CPP_DIR:?Set ARTSYN_GODOT_CPP_DIR to your godot-cpp checkout}"

require_torch_root=true
case "${ENABLE_LIBTORCH_BACKEND^^}" in
    OFF|FALSE|0|NO)
        require_torch_root=false
        ;;
esac

if [[ "$require_torch_root" == true ]]; then
    : "${ARTSYN_TORCH_ROOT:?Set ARTSYN_TORCH_ROOT to your LibTorch root}"
fi

case "$BUILD_TYPE" in
    Debug)
        GODOT_CPP_LIBRARY_DEFAULT="$ARTSYN_GODOT_CPP_DIR/bin/libgodot-cpp.linux.template_debug.x86_64.a"
        ;;
    Release|RelWithDebInfo|MinSizeRel)
        GODOT_CPP_LIBRARY_DEFAULT="$ARTSYN_GODOT_CPP_DIR/bin/libgodot-cpp.linux.template_release.x86_64.a"
        ;;
    *)
        echo "Unsupported ARTSYN_BUILD_TYPE: $BUILD_TYPE" >&2
        exit 1
        ;;
esac

ARTSYN_GODOT_CPP_LIBRARY="${ARTSYN_GODOT_CPP_LIBRARY:-$GODOT_CPP_LIBRARY_DEFAULT}"

if [[ ! -f "$ARTSYN_GODOT_CPP_LIBRARY" ]]; then
    echo "Missing godot-cpp static library for build type $BUILD_TYPE:" >&2
    echo "  $ARTSYN_GODOT_CPP_LIBRARY" >&2
    echo "Set ARTSYN_GODOT_CPP_LIBRARY explicitly if your file is elsewhere." >&2
    exit 1
fi

cmake_args=(
    -S "$ROOT_DIR"
    -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DGODOT_CPP_DIR="$ARTSYN_GODOT_CPP_DIR"
    -DGODOT_CPP_LIBRARY="$ARTSYN_GODOT_CPP_LIBRARY"
    -DARTSYN_ENABLE_LIBTORCH_BACKEND="$ENABLE_LIBTORCH_BACKEND"
)

if [[ -n "${ARTSYN_TORCH_ROOT:-}" ]]; then
    cmake_args+=(-DARTSYN_TORCH_ROOT="$ARTSYN_TORCH_ROOT")
fi

if [[ -n "$OUTPUT_DIR" ]]; then
    cmake_args+=(-DARTSYN_OUTPUT_DIR="$OUTPUT_DIR")
fi

if [[ -n "$EXTENSION_STEM" ]]; then
    cmake_args+=(-DARTSYN_EXTENSION_STEM="$EXTENSION_STEM")
fi

if [[ -n "$BUNDLE_TORCH_LIBS" ]]; then
    cmake_args+=(-DARTSYN_BUNDLE_TORCH_LIBS="$BUNDLE_TORCH_LIBS")
fi

if [[ -n "$BUNDLE_CUDA_RUNTIME" ]]; then
    cmake_args+=(-DARTSYN_BUNDLE_CUDA_RUNTIME="$BUNDLE_CUDA_RUNTIME")
fi

if [[ -n "$ENABLE_ONNX_RUNTIME" ]]; then
    cmake_args+=(-DARTSYN_ENABLE_ONNX_RUNTIME="$ENABLE_ONNX_RUNTIME")
fi

if [[ -n "$BUNDLE_ONNX_RUNTIME" ]]; then
    cmake_args+=(-DARTSYN_BUNDLE_ONNX_RUNTIME="$BUNDLE_ONNX_RUNTIME")
fi

if [[ -n "$ONNX_RUNTIME_ROOT" ]]; then
    cmake_args+=(-DARTSYN_ONNX_RUNTIME_ROOT="$ONNX_RUNTIME_ROOT")
fi

if [[ -n "$ONNX_RUNTIME_INCLUDE_DIR" ]]; then
    cmake_args+=(-DARTSYN_ONNX_RUNTIME_INCLUDE_DIR="$ONNX_RUNTIME_INCLUDE_DIR")
fi

if [[ -n "$ONNX_RUNTIME_LIBRARY" ]]; then
    cmake_args+=(-DARTSYN_ONNX_RUNTIME_LIBRARY="$ONNX_RUNTIME_LIBRARY")
fi

if [[ -n "$EXTRA_RUNTIME_LIBRARY_DIRS" ]]; then
    cmake_args+=(-DARTSYN_EXTRA_RUNTIME_LIBRARY_DIRS="$EXTRA_RUNTIME_LIBRARY_DIRS")
fi

cmake "${cmake_args[@]}"

cmake --build "$BUILD_DIR"
