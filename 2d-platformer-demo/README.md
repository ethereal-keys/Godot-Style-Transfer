# ArtSyn Demo Project

This folder is the usage example for the ArtSyn plugin.

It is a Godot 4 platformer demo wired to show real-time style transfer on a `SubViewport` through `MyCustomNode`. The project is useful for:

- validating packaged addon builds
- checking runtime provider selection in a real scene
- demonstrating how to connect `viewport_path` and `display_sprite_path`
- profiling `inference_scale` and other runtime settings

## Key Files

- `project.godot`
  demo project entry point
- `game_viewport.tscn`
  main ArtSyn example scene
- `my_custom_node.gd`
  simple scene-side wrapper around `MyCustomNode`
- `addons/artsyn/`
  project-facing addon layout used for local testing and packaging

## What This Project Should Represent

This folder is intentionally the example consumer of the plugin, not the source of truth for the native extension itself.

- plugin source belongs in `ArtSyn_extension/`
- model training and export belong in `Real-Time-Style-Transfer/`
- packaged addon/runtime combinations should be tested here

## Running The Demo

1. Make sure `addons/artsyn/` contains the addon files and a matching runtime `bin/` folder.
2. Open this folder as a Godot project.
3. Run `game_viewport.tscn`.
4. Watch the startup log for the selected execution provider.

Expected examples:

```text
Loaded model ... using backend ONNX Runtime [CPUExecutionProvider]
Loaded model ... using backend ONNX Runtime [CUDAExecutionProvider]
```

## Notes

- The current demo defaults to the ONNX model under `addons/artsyn/models/model.onnx`.
- This project may regenerate `.godot/` editor files locally; those are not meant to be tracked.
