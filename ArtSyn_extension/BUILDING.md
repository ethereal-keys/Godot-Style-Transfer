# Building ArtSyn_extension

The supported build path is `CMake`.

## Requirements

- a `godot-cpp` checkout with generated bindings and a built static library
- a LibTorch installation for builds that include the LibTorch backend
- CMake 3.18+
- a C++17 compiler

## Dependency Inputs

The build accepts dependency locations in either CMake cache variables or environment variables.

Supported inputs:

- `GODOT_CPP_DIR` or `ARTSYN_GODOT_CPP_DIR`
- `GODOT_CPP_LIBRARY` or `ARTSYN_GODOT_CPP_LIBRARY`
- `ARTSYN_TORCH_ROOT`
- `Torch_DIR` or `ARTSYN_TORCH_DIR`
- `ARTSYN_BUILD_TYPE`
- `ARTSYN_OUTPUT_DIR`
- `ARTSYN_EXTENSION_STEM`
- `ARTSYN_ENABLE_LIBTORCH_BACKEND`
- `ARTSYN_BUNDLE_TORCH_LIBS`
- `ARTSYN_BUNDLE_CUDA_RUNTIME`
- `ARTSYN_ENABLE_ONNX_RUNTIME`
- `ARTSYN_BUNDLE_ONNX_RUNTIME`
- `ARTSYN_ONNX_RUNTIME_ROOT`
- `ARTSYN_ONNX_RUNTIME_INCLUDE_DIR`
- `ARTSYN_ONNX_RUNTIME_LIBRARY`
- `ARTSYN_EXTRA_RUNTIME_LIBRARY_DIRS`

`ARTSYN_TORCH_ROOT` should point at the LibTorch root directory, for example the folder that contains `share/cmake/Torch/TorchConfig.cmake`.

## Configure And Build

Example using explicit CMake variables:

```bash
cmake -S ArtSyn_extension -B ArtSyn_extension/build/linux-debug \
  -DGODOT_CPP_DIR=/path/to/godot-cpp \
  -DGODOT_CPP_LIBRARY=/path/to/libgodot-cpp.linux.template_debug.x86_64.a \
  -DARTSYN_TORCH_ROOT=/path/to/libtorch

cmake --build ArtSyn_extension/build/linux-debug
```

Example using environment variables:

```bash
export ARTSYN_GODOT_CPP_DIR=/path/to/godot-cpp
export ARTSYN_GODOT_CPP_LIBRARY=/path/to/libgodot-cpp.linux.template_debug.x86_64.a
export ARTSYN_TORCH_ROOT=/path/to/libtorch

cmake -S ArtSyn_extension -B ArtSyn_extension/build/linux-debug
cmake --build ArtSyn_extension/build/linux-debug
```

Or use the helper script:

```bash
export ARTSYN_GODOT_CPP_DIR=/path/to/godot-cpp
export ARTSYN_GODOT_CPP_LIBRARY=/path/to/libgodot-cpp.linux.template_debug.x86_64.a
export ARTSYN_TORCH_ROOT=/path/to/libtorch

./ArtSyn_extension/build_extension.sh
```

Build a release variant with:

```bash
export ARTSYN_BUILD_TYPE=Release
./ArtSyn_extension/build_extension.sh
```

The helper script expects a matching `godot-cpp` static library for the selected build type:

- `Debug` -> `libgodot-cpp.linux.template_debug.x86_64.a`
- `Release` -> `libgodot-cpp.linux.template_release.x86_64.a`

## Optional ONNX Runtime Integration

The `onnx` backend id is always available at the Godot API layer, but it only performs real inference when the extension is built with ONNX Runtime enabled.

Enable that build path with:

```bash
export ARTSYN_ENABLE_ONNX_RUNTIME=ON
export ARTSYN_ONNX_RUNTIME_ROOT=/path/to/onnxruntime

./ArtSyn_extension/build_extension.sh
```

Or pass the resolved paths explicitly:

```bash
cmake -S ArtSyn_extension -B ArtSyn_extension/build/linux-debug \
  -DGODOT_CPP_DIR=/path/to/godot-cpp \
  -DGODOT_CPP_LIBRARY=/path/to/libgodot-cpp.linux.template_debug.x86_64.a \
  -DARTSYN_TORCH_ROOT=/path/to/libtorch \
  -DARTSYN_ENABLE_ONNX_RUNTIME=ON \
  -DARTSYN_ONNX_RUNTIME_INCLUDE_DIR=/path/to/onnxruntime/include \
  -DARTSYN_ONNX_RUNTIME_LIBRARY=/path/to/onnxruntime/lib/libonnxruntime.so
```

If `ARTSYN_ENABLE_ONNX_RUNTIME=ON` is set without valid headers and libraries, CMake now fails fast with a setup error instead of compiling a broken backend.

The ONNX backend now auto-selects `CUDAExecutionProvider` when the supplied ONNX Runtime build exposes it. If the SDK only exposes `CPUExecutionProvider`, the backend cleanly falls back to CPU.

For NVIDIA GPU inference, use a GPU-enabled ONNX Runtime distribution whose `lib/` folder includes `libonnxruntime_providers_cuda.so`.

## ONNX-Only Stub Builds

The extension can now be built without LibTorch at all. This is useful for validating the smaller-addon direction before a full ONNX Runtime deployment is wired up.

Example:

```bash
export ARTSYN_ENABLE_LIBTORCH_BACKEND=OFF
./ArtSyn_extension/build_extension.sh
```

In that mode:

- LibTorch is not discovered or linked
- `backend_id = "libtorch"` is unavailable
- `backend_id = "onnx"` remains available as the fallback backend path
- the node defaults to the ONNX backend and `res://addons/artsyn/models/model.onnx`

## Output

By default, the built extension is written to:

`2d-platformer-demo/addons/artsyn/bin`

The default file names are:

- debug: `libmy_extension.debug.so`
- release: `libmy_extension.release.so`

Override this with:

```bash
-DARTSYN_OUTPUT_DIR=/custom/output/path
```

## Current Notes

- the default output target is the demo project's addon folder
- the current default `godot-cpp` library name assumes Linux debug `x86_64`
- the build currently bundles detected runtime dependencies beside the extension by default
- ONNX Runtime provider libraries such as `libonnxruntime_providers_cuda.so` are now bundled when present
- the runtime now selects inference backends through a small factory seam
- `libtorch` is the only fully validated backend today
- `onnx` can be compiled as a real backend when ONNX Runtime is provided, otherwise it returns a clear unsupported error
- ONNX runtime logs now report the selected execution provider at model load time
- the LibTorch backend can now be compiled out entirely for smaller experimental builds
- broader platform and build-type handling will be expanded as packaging work continues

## Packaging

After a successful build, create a distributable addon archive with:

```bash
./ArtSyn_extension/package_addon.sh
```

Override the archive name or destination with:

- `ARTSYN_PACKAGE_NAME`
- `ARTSYN_DIST_DIR`
- `ARTSYN_ADDON_DIR`
- `ARTSYN_EXTENSION_STEM`
- `ARTSYN_PACKAGE_RUNTIME_PROFILE`
- `ARTSYN_DRY_RUN=1`

Split packaging is also supported:

```bash
ARTSYN_PACKAGE_MODE=core ./ArtSyn_extension/package_addon.sh
ARTSYN_PACKAGE_MODE=runtime ./ArtSyn_extension/package_addon.sh
```

Supported package modes:

- `full`
- `core`
- `runtime`

The package script inspects the addon `bin/` directory and chooses a runtime profile automatically:

- ONNX Runtime libraries present without LibTorch -> `onnxruntime`
- ONNX Runtime CUDA provider library present without LibTorch -> `onnxruntime-cuda`
- LibTorch or `libc10` libraries present -> `nvidia-libtorch`
- neither detected -> `native`

Default package names are then built from that profile:

- both debug and release extension binaries -> `artsyn-linux-x86_64-<profile>.zip`
- only debug -> `artsyn-linux-x86_64-<profile>-debug.zip`
- only release -> `artsyn-linux-x86_64-<profile>-release.zip`

The same suffixing is used for `runtime` packages.

Runtime-only packages now include the addon `README.md` alongside the `bin/` directory so users have install guidance even when downloading a backend pack separately.

Use `ARTSYN_DRY_RUN=1` to inspect the resolved package name and staged files without creating a zip archive.
