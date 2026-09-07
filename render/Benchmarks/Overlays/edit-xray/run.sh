#!/bin/bash
# Load this scene in the app. Run from the repository root.
exec ./build/MeshEditor res/benchmarks/Overlays/Overlays.gltf --headless --quiet --frames 2 --screenshot render/Benchmarks/Overlays/edit-xray/edit-xray.webp --shading solid --overlays --selection-xray --edit vertex --bench-action box-select-orbit --camera Axes
