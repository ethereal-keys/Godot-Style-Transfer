# ArtSyn For Godot

ArtSyn is a Godot 4 addon for real-time viewport style transfer.

The addon ships with a default style-transfer model under `addons/artsyn/models/`, so users do not need to download a model separately for the first run.

## Current Support

- Godot `4.2+`
- Linux `x86_64`
- ONNX Runtime backend
- two runtime distribution options:
  - `onnxruntime`: smaller CPU-only package, slower
  - `onnxruntime-cuda`: much larger NVIDIA package, faster

## Install

1. Copy `addons/artsyn/` into your Godot project.
2. Make sure the addon contains:
   - `ArtSyn_extension.gdextension`
   - `models/model.onnx`
   - `bin/libmy_extension.<variant>.so`
3. Open the project in Godot.
4. Add a `MyCustomNode` to the scene.
5. Set:
   - `viewport_path` to the `SubViewport` you want to stylize
   - `display_sprite_path` to the `Sprite2D` that should display the stylized output
6. Leave the defaults unless you want to override them:
   - `backend_id = "onnx"`
   - `model_path = "res://addons/artsyn/models/model.onnx"`

## Bundled Model

- `models/model.onnx` is the default runtime model for the ONNX backend.
- `models/model.pth` is kept for LibTorch compatibility and backend comparisons.
- Advanced users can point `model_path` at another compatible `.onnx` or `.pth` file.

## Runtime Options

### CPU Runtime Pack

Use the CPU pack when you want the smallest install and do not mind lower framerates.

Expected startup log:

```text
Loaded model ... using backend ONNX Runtime [CPUExecutionProvider]
```

### NVIDIA GPU Runtime Pack

Use the CUDA pack when you want the fastest current runtime on Linux.

Expected startup log:

```text
Loaded model ... using backend ONNX Runtime [CUDAExecutionProvider]
```

This pack is much larger because it bundles the ONNX Runtime CUDA provider and NVIDIA runtime libraries.

## Tuning

- `inference_scale`
  Lowers internal inference resolution for better framerate at the cost of image sharpness.
- `benchmark_logging_enabled`
  Prints average FPS every few seconds for profiling.
- `verbose_logging_enabled`
  Enables extra setup logs while debugging scene wiring.

## Troubleshooting

- If the addon loads but reports `CPUExecutionProvider`, you are using the CPU runtime pack or GPU initialization failed.
- If the log mentions missing CUDA libraries such as `libcudnn.so`, the GPU runtime pack is incomplete or not the active `bin/` folder.
- On hybrid laptops, Godot may need to be launched with NVIDIA offload environment variables for the CUDA provider to see the discrete GPU.
- If the node reports that the model is not loaded, confirm `model_path` exists inside your project and matches the selected backend.

## Current Limitations

- Linux-only for now
- the shipped ONNX model is currently fixed-size, so the addon resizes input images before inference
- the current frame pipeline still reads the viewport into CPU memory and uploads the processed image back to a texture

## Repo Docs

- root project guide: `README.md`
- build and packaging guide: `ArtSyn_extension/BUILDING.md`
- release planning roadmap: `docs/distribution-roadmap.md`
