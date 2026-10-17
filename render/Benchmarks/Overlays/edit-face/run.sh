#!/bin/bash
# Run from the repository root.
exec ./build/MeshEditor res/benchmarks/Overlays/Overlays.gltf --headless --frames 2 --screenshot render/Benchmarks/Overlays/edit-face/edit-face.webp --shading solid --overlays --edit face --bench-action box-select-orbit --camera Axes
