#!/bin/bash
# Run from the repository root.
exec ./build/MeshEditor res/benchmarks/Overlays/Overlays.gltf --headless --frames 12 --screenshot render/Benchmarks/Overlays/overview/overview.webp --shading solid --overlays --select-all --play --camera Overview --display bounds,face-normals,vertex-normals
