#!/bin/bash
# Run from the repository root.
exec ./build/MeshEditor res/benchmarks/Overlays/Overlays.gltf --headless --frames 1 --screenshot render/Benchmarks/Overlays/pick-1/pick-1.webp --shading preview --overlays --bench-action pick-cycle --camera Pick
