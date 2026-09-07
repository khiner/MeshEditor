#!/bin/bash
# Load this scene in the app. Run from the repository root.
exec ./build/MeshEditor res/benchmarks/Overlays/Overlays.gltf --headless --quiet --frames 1 --screenshot render/Benchmarks/Overlays/axes/axes.webp --shading solid --overlays --select-all --bench-action steady --camera Axes
