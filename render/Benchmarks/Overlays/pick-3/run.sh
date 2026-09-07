#!/bin/bash
# Load this scene in the app. Run from the repository root.
exec ./build/MeshEditor res/benchmarks/Overlays/Overlays.gltf --headless --quiet --frames 3 --screenshot render/Benchmarks/Overlays/pick-3/pick-3.webp --shading preview --overlays --bench-action pick-cycle --camera Pick
