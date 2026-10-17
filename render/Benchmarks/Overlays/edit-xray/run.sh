#!/bin/bash
# Run from the repository root.
exec ./build/MeshEditor res/benchmarks/Overlays/Overlays.gltf --headless --frames 2 --screenshot render/Benchmarks/Overlays/edit-xray/edit-xray.webp --shading solid --overlays --xray --edit vertex --bench-action box-select-orbit --camera Axes
