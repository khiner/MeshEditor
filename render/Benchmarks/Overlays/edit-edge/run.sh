#!/bin/bash
# Load this scene in the app. Run from the repository root.
exec ./build/MeshEditor res/benchmarks/Overlays/Overlays.gltf --headless --quiet --frames 2 --screenshot render/Benchmarks/Overlays/edit-edge/edit-edge.webp --shading solid --overlays --edit edge --bench-action box-select-orbit --camera Axes
