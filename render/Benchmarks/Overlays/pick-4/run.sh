#!/bin/bash
# Load this scene in the app. Run from the repository root.
exec ./build/MeshEditor res/benchmarks/Overlays/Overlays.gltf --headless --quiet --frames 4 --screenshot render/Benchmarks/Overlays/pick-4/pick-4.webp --shading preview --overlays --bench-action pick-cycle --camera Pick
