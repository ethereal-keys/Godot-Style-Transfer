# Install And Release Guide

This guide is the practical bridge between the current repo and a user-facing plugin release.

## Package Types

ArtSyn is currently easiest to distribute as two layers:

- `artsyn-core`
  - addon metadata
  - scripts
  - bundled models
  - docs
- runtime pack
  - native extension library
  - backend runtime libraries

That keeps the core addon stable while letting users choose the runtime that matches their machine.

## Current Runtime Choices

### `onnxruntime`

- smallest current runtime pack
- CPU inference only
- easiest install story
- slower in real-time scenes

### `onnxruntime-cuda`

- larger runtime pack
- NVIDIA GPU acceleration through ONNX Runtime CUDA
- best current performance path on Linux
- requires a working NVIDIA driver stack

## What Users Need

For the first public posts, the user install story should be:

1. Download `artsyn-core`.
2. Download one runtime pack:
   - CPU for convenience
   - CUDA for performance
3. Merge both into `addons/artsyn/` inside a Godot project.
4. Open Godot and add `MyCustomNode`.
5. Set:
   - `viewport_path`
   - `display_sprite_path`
6. Keep the default ONNX model unless they want to experiment.

## Building Release Artifacts

Build the native extension first.

Example CPU-oriented ONNX build:

```bash
export ARTSYN_GODOT_CPP_DIR=/path/to/godot-cpp
export ARTSYN_GODOT_CPP_LIBRARY=/path/to/libgodot-cpp.linux.template_release.x86_64.a
export ARTSYN_BUILD_TYPE=Release
export ARTSYN_ENABLE_LIBTORCH_BACKEND=OFF
export ARTSYN_ENABLE_ONNX_RUNTIME=ON
export ARTSYN_ONNX_RUNTIME_ROOT=/path/to/onnxruntime

./ArtSyn_extension/build_extension.sh
```

Example CUDA-oriented ONNX build:

```bash
export ARTSYN_GODOT_CPP_DIR=/path/to/godot-cpp
export ARTSYN_GODOT_CPP_LIBRARY=/path/to/libgodot-cpp.linux.template_release.x86_64.a
export ARTSYN_BUILD_TYPE=Release
export ARTSYN_ENABLE_LIBTORCH_BACKEND=OFF
export ARTSYN_ENABLE_ONNX_RUNTIME=ON
export ARTSYN_ONNX_RUNTIME_ROOT=/path/to/onnxruntime-gpu
export ARTSYN_EXTRA_RUNTIME_LIBRARY_DIRS=/path/to/cuda/libs

./ArtSyn_extension/build_extension.sh
```

## Creating Packages

Core package:

```bash
ARTSYN_PACKAGE_MODE=core ./ArtSyn_extension/package_addon.sh
```

Full package:

```bash
./ArtSyn_extension/package_addon.sh
```

Runtime-only package:

```bash
ARTSYN_PACKAGE_MODE=runtime ./ArtSyn_extension/package_addon.sh
```

Dry-run package inspection:

```bash
ARTSYN_DRY_RUN=1 ./ArtSyn_extension/package_addon.sh
```

## Recommended First Public Support Matrix

Keep this narrow at first:

- OS: Linux `x86_64`
- Engine: Godot `4.2+`
- backend: ONNX Runtime
- runtime packs:
  - CPU
  - NVIDIA CUDA

## Recommended Release Checklist

- build debug and release binaries successfully
- verify the addon loads from packaged artifacts only
- test CPU package on a clean machine
- test CUDA package on a clean NVIDIA Linux machine
- confirm startup log shows the expected execution provider
- confirm `models/model.onnx` is bundled in the core package
- confirm the demo scene works without local path edits
- update release notes with known limitations

## Known User-Facing Caveats

- the CUDA package is much larger than the CPU package
- the shipped ONNX model currently uses fixed input dimensions
- the current frame loop still uses CPU image readback and texture upload
- hybrid laptops may need explicit NVIDIA offload environment variables

## Messaging For Posts

The current honest positioning is:

- real-time style transfer addon for Godot 4
- bundled default model
- Linux-first release
- CPU and NVIDIA runtime options
- still improving performance and package size over time
