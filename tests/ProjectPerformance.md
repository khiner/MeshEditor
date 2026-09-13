# Project history measurements

Run the Metal-buffer and mesh-edit benchmarks:

```sh
cmake --build build --target MeshEditorProjectBench -j 6
./build/tests/MeshEditorProjectBench
```

Measured on an Apple M5 Max with macOS 26.6.2 and Homebrew Clang 23.1.0.
The Debug build uses `-O2` for the store, Metal, mesh, renderer, and benchmark, and `-O0` for editor orchestration.
Each table entry is the median of three process medians with warm filesystem caches.
Each process measures 30 edits and 29 cold neighboring restores.
Audits and replay checks run outside the timers.

## Sparse buffer edits

Initialize each 4 KiB page with a distinct eight-byte value, then edit eight bytes in the first page.
Commit includes capture, hashing, manifest updates, and write enqueueing.
Flush writes before measuring navigation.
These measurements cover buffer-history scaling independently of scene processing and rendering.

| Live buffer | Write + commit | Hot neighbor | Cold neighbor | Content appended/edit |
|---|---:|---:|---:|---:|
| 1 MiB | 0.0067 ms | 0.0010 ms | 0.0431 ms | 5,438 B |
| 16 MiB | 0.0069 ms | 0.0010 ms | 0.0434 ms | 6,458 B |
| 64 MiB | 0.0068 ms | 0.0010 ms | 0.0409 ms | 6,509 B |

After eviction at the root, copied payload size is zero.
Estimated retained history memory is 0.291, 0.636, and 1.738 MiB respectively, excluding live buffers.
Peak queued and staged writes were 1.019, 13.248, and 16.150 MiB.
Batching reduced the 64 MiB case from approximately 65 MiB of pending writes.
The history payload cap excludes write buffers and metadata.

## Mesh editing

Translate all selected vertices of a UV sphere through a staged gizmo action and gesture completion.
Restores include `Project::AfterRestore` and `ProcessComponentEvents`.
Rendered rows also submit and wait for the 128 x 128 viewport after each operation.

| Vertices | Render included | Drag + commit | Hot neighbor | Cold neighbor |
|---|---|---:|---:|---:|
| 1,986 | No | 0.363 ms | 0.275 ms | 0.333 ms |
| 32,514 | No | 0.728 ms | 0.516 ms | 0.715 ms |
| 1,986 | Yes | 0.936 ms | 0.829 ms | 0.878 ms |
| 32,514 | Yes | 3.470 ms | 3.203 ms | 3.440 ms |

An earlier implementation rebuilt meshlets and LODs after position-only restoration.
For 32,514 vertices without rendering, a development run measured 31.477 ms hot and 32.423 ms cold.
Updating only restored vertex ranges reduced those times to 0.493 ms and 0.703 ms in a matching run.
These comparisons use single-process medians, separate from the repeated measurements above.
Topology changes and geometry outside active edit meshes still require meshlet and LOD rebuilding.

`MeshEditorProjectTest` checks Persistent state, surface rendering, and replay after dense and single-vertex undo, redo, and cold restoration.

## History window

A development probe measured a 400 x 300 ImGui window with 10,001 states over 100 idle frames.
Caching tree order and clipping invisible rows reduced the median from 2.655 ms to 0.005 ms.
Both measurements are single-process results from implementations of the new History window.
