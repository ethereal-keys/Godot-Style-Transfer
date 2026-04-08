# Distribution And Runtime Roadmap

## North Star

Turn ArtSyn into a normal Godot addon with:

- a small core download
- prebuilt native backend packs
- no local compiler or SDK setup for end users
- a runtime abstraction so inference backends can be swapped
- a credible path toward 60 FPS on supported hardware

## Current Reality

The current prototype is promising, but it is not distributable yet:

- the extension build depends on local machine paths
- the model load path is hardcoded to a local filesystem path
- the shipped binary is Linux-only and tied to a local LibTorch install
- the current frame loop moves image data through CPU memory every frame

Those issues make the current addon large, fragile, and hard to install on other systems.

## Strategy

Use a phased approach:

1. Stabilize the current prototype and make it reproducible.
2. Ship a narrow but real first target: Linux `x86_64` + NVIDIA GPU.
3. Benchmark the current architecture honestly before over-investing in packaging.
4. Migrate away from LibTorch in the shipped addon to improve size and portability.
5. Split the product into a small core addon plus backend packs.

## Backend Direction

Recommended direction:

- short term: keep LibTorch only as a bridge for the first Linux/NVIDIA alpha
- medium term: target ONNX Runtime as the first replacement backend
- fallback option: evaluate NCNN if ONNX Runtime is still too large or awkward to ship

Why this direction:

- the training pipeline already exports TorchScript
- the repo already has an ONNX export path
- ONNX gives a better packaging story than shipping LibTorch directly

## Phase 0: Stabilize The Prototype

Target: 1-2 weeks

Goals:

- make the project buildable in a repeatable way
- remove machine-specific assumptions
- create clean seams for later backend swaps
- measure real performance without noisy logging

Tasks:

- choose one build system and standardize on it
  - recommended: `CMake`
- restructure the native addon into a normal `addons/artsyn/` layout
- remove all absolute paths from source and build scripts
- make the model path project-relative and configurable
- add a small C++ backend interface such as `StyleTransferBackend`
- keep the first implementation behind that interface as LibTorch
- remove per-frame debug printing from the hot path
- add benchmark mode and timing instrumentation
- add an inference resolution setting for controlled performance testing
- define the first supported target clearly
  - Linux `x86_64`
  - Godot 4.x
  - NVIDIA GPU
  - one tested CUDA/runtime combination

Deliverables:

- reproducible local build
- clean addon folder structure
- repeatable FPS and latency measurements

## Phase 1: Linux/NVIDIA Alpha

Target: 2-4 weeks

Goals:

- ship the first genuinely installable addon
- keep the support matrix narrow and explicit
- avoid asking users to build anything themselves

Tasks:

- make the CUDA build reproducible and relocatable
- package only the required runtime libraries
- ship both debug and release native binaries
- update the `.gdextension` manifest for supported binaries
- bundle a default model with relative paths
- add clear backend availability checks and user-facing errors
- create a zip release that can be dropped into a Godot project
- add CI for Linux build and smoke-test loading

Deliverables:

- installable Linux/NVIDIA alpha release
- simple install instructions
- validated demo project using shipped artifacts only

Decision gate:

- if the current architecture gets close enough to target performance, keep using LibTorch as the bridge a little longer
- if not, stop polishing LibTorch packaging and move faster to backend migration

## Parallel Track: Performance Hardening

This runs alongside Phase 1.

Reason:

Packaging alone will not guarantee 60 FPS. The current loop reads the viewport into CPU memory, performs inference, then uploads the result back into a texture. Even with CUDA, that transfer pattern may be the real bottleneck.

Tasks:

- benchmark inference time separately from transfer time
- benchmark multiple internal inference resolutions
- cache reusable buffers and avoid per-frame allocation churn
- remove logging and other debug-only costs from the frame loop
- document target hardware and measured results

Deliverables:

- performance baseline table
- clear view of whether the current architecture can scale

## Phase 2: Backend Migration Spike

Target: 2-3 weeks

Goals:

- reduce download size
- improve installation story
- open a better path to cross-platform support

Tasks:

- export a production ONNX model from the current training pipeline
- update export tooling if fixed or dynamic shapes need to be standardized
- implement an `OnnxBackend` behind the backend interface
  - first milestone: land a compile-time scaffold and selectable backend id before wiring ONNX Runtime itself
  - second milestone: make ONNX Runtime integration optional at build time so the repo can support both stub and real ONNX builds
  - third milestone: allow ONNX-only builds that compile without LibTorch at all
- add output-parity tests between LibTorch and ONNX
- compare package size, startup time, FPS, and deployment complexity

Decision gate:

- if ONNX Runtime is fast enough and simpler to ship, make it the default backend path
- if ONNX Runtime is still too heavy, run an NCNN spike next

Deliverables:

- working ONNX backend prototype
- comparison report against LibTorch
- clear recommendation for the default shipped backend

## Phase 3: Medium-Term Productization

Target: 4-8 weeks

Goals:

- make the default download small
- make installs predictable
- support more than one platform without a monolithic package

Tasks:

- split releases into:
  - core addon
  - backend packs
- keep only scripts, scenes, metadata, config, and default model in the core addon
- put native binaries and heavier runtime dependencies in backend packs
- add backend selection and capability detection in addon settings
- build a CI matrix for Linux and Windows first
- keep LibTorch only as an internal validation backend if still useful
- write user-focused installation and support docs

Example backend packs:

- `linux-x86_64-nvidia`
- `windows-x86_64-nvidia`
- optional `cpu-fallback`

Deliverables:

- small core addon download
- optional platform/backend packs
- first real cross-platform release story

## Immediate Work Queue

These are the first fixes we should tackle together:

1. Standardize on `CMake`.
2. Move the addon into `addons/artsyn/`.
3. Replace hardcoded model and library paths.
4. Add a backend interface around the current LibTorch code.
5. Remove per-frame FPS logging from the hot loop.
6. Add benchmark mode and inference resolution control.
7. Make a narrow Linux/NVIDIA support matrix explicit in docs and code.

## Success Criteria

Short term success:

- another Linux/NVIDIA user can install the addon from a release zip without editing source paths
- the addon loads without requiring a local LibTorch development setup
- we have trustworthy performance numbers on supported hardware

Medium-term success:

- the default addon download is much smaller than a LibTorch-based bundle
- backend/runtime choice is abstracted behind a stable interface
- Linux and Windows distribution are both realistic without source builds

## Notes

- CPU-only support should remain optional, not the main product direction.
- Cross-platform support should not be attempted all at once.
- Performance validation should guide packaging investment, not the other way around.
