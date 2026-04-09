# Godot Style Transfer

ArtSyn is a Godot 4 GDExtension addon for real-time style transfer on viewport output.

## Demo

<p>
  <a href="https://youtu.be/ad-yBz3aguA">
    <img src="docs/assets/youtube-demo-button.svg" alt="Watch the ArtSyn demo on YouTube" width="360">
  </a>
  <a href="https://sushanth-kashyap.vercel.app/projects/style-transfer">
    <img src="docs/assets/writeup-button.svg" alt="Read the ArtSyn project write-up" width="360">
  </a>
</p>

Demo video: [YouTube](https://youtu.be/ad-yBz3aguA) | Write-up: [Project notes](https://sushanth-kashyap.vercel.app/projects/style-transfer)

The write-up is still a good retrospective on how the project came together, but the repo docs below are the current source of truth for the addon layout, build flow, and packaging.

The repo now contains both:

- the native extension source in `ArtSyn_extension/`
- a project-facing addon layout in `2d-platformer-demo/addons/artsyn/`

## Docs

- [Addon README](2d-platformer-demo/addons/artsyn/README.md)
- [Build Guide](ArtSyn_extension/BUILDING.md)
- [Install And Release Guide](docs/install-and-release-guide.md)
- [Distribution Roadmap](docs/distribution-roadmap.md)

## Repo Roles

The repo is organized around three main surfaces:

- `ArtSyn_extension/`
  plugin source, build logic, and packaging scripts
- `2d-platformer-demo/`
  usage example project for testing and showcasing the addon
- `Real-Time-Style-Transfer/`
  training and export pipeline for making new models

## What Works Today

- Linux `x86_64`
- Godot `4.2+`
- ONNX Runtime backend
- bundled default model inside the addon
- CPU and NVIDIA CUDA runtime packaging paths

This is still an early release candidate, but it is no longer just a machine-local prototype.

## Quick Start

If you just want to try the addon in Godot:

1. Put `addons/artsyn/` into your project.
2. Make sure the addon contains a matching runtime `bin/` folder.
3. Open the project in Godot.
4. Add `MyCustomNode` to your scene.
5. Set:
   - `viewport_path`
   - `display_sprite_path`
6. Keep the defaults for the ONNX path:
   - `backend_id = "onnx"`
   - `model_path = "res://addons/artsyn/models/model.onnx"`

The default model is bundled with the addon, so users do not need to fetch a separate model for the first install.

## Runtime Packages

Current distribution strategy:

- `artsyn-core`
  addon metadata, scripts, models, and docs
- `artsyn-linux-x86_64-onnxruntime`
  smaller CPU-oriented runtime pack
- `artsyn-linux-x86_64-onnxruntime-cuda`
  larger NVIDIA runtime pack with better performance

This lets users choose between smaller downloads and better GPU acceleration.

## Repo Layout

- `ArtSyn_extension/`
  native extension source, build scripts, and packaging scripts
- `2d-platformer-demo/`
  example Godot project that consumes the addon
- `2d-platformer-demo/addons/artsyn/`
  project-facing addon layout intended for distribution
- `Real-Time-Style-Transfer/`
  training and export code for the style-transfer model

## Credits

- Model/training base repo: <https://github.com/1627180283/real-time-Style-Transfer>
