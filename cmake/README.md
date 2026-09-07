# Production libraries

`Libraries.cmake` assigns each production source to one static library.
The app and test executables link these libraries, so building both compiles each production source once per configuration.

| Library | Owns | Debug optimization |
| --- | --- | --- |
| MeshEditorMetal | Metal resources, shader compilation, command submission primitives | `-O2` |
| MeshEditorMesh | Mesh storage, topology, CPU geometry, BVH, mesh compute pipelines | `-O2` |
| MeshEditorScene | Entities, hierarchy, transforms, armatures, camera and selection state | `-O2` |
| MeshEditorRender | GPU scene preparation, render pipelines, selection rendering, frame submission | `-O2` |
| MeshEditorPhysics | Jolt integration, colliders and contact collection | `-O2` |
| MeshEditorAudio | Audio devices, decoding, DSP, contact models and modal solves | `-O2` |
| MeshEditorAssets | glTF, OBJ, PLY, material and sample import | `-O0` |
| MeshEditorEditor | Actions, replay, snapshots, reactive scene orchestration and domain integration | `-O0` with the audio exceptions below |
| MeshEditorUi | Panels, widgets, plots, presentation and ImGui platform integration | `-O0` |
| MeshEditorPlatform | Native windows, events and file dialogs | `-O0` |

`main.cpp` owns application startup and the event loop.
It uses the cold optimization policy.
`AudioIntegration.cpp` and `SurfaceAudio.cpp` stay at `-O2` because they run contact-processing and modal-input numerical loops.
Release builds optimize all production libraries at `-O2`.
Bundled libraries and image codecs retain their optimized builds.
Debug builds retain debug information and assertions.

## Dependencies

Metal is the base, Mesh depends on Metal, and Scene depends on Mesh.
Render, Physics and Audio depend on Scene.
Assets uses Render and Audio to materialize imported resources.
Editor composes Assets, Physics and Audio.
Platform depends on Metal, and Ui composes Editor and Platform.
Dependencies must not point back toward their callers.

Keep ImGui and ImPlot inside Ui and the app.
Keep file-format parsers inside Assets and serialization templates inside their owning implementation files or serialization headers.
Keep numerical work out of UI and action handlers.
Snapshot registration is split by domain under `src/snapshot`, with one shared encoding implementation.
Mesh and Render use the same Metal buffer context and shared storage.

Small internal targets own common file/path/compression helpers, image codecs, ImGui and ImPlot.
They do not introduce alternate scene models or storage.
Public target dependencies carry headers required by consumers; implementation dependencies stay private.

## Building individual targets

Configure with the normal project options, then select the library, app or test target:

```sh
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target MeshEditor
cmake --build build --target MeshEditorTests
cmake --build build --target MeshEditorAudio
```

Use `-DMESHEDITOR_SURFACE_AUDIO=ON` in a separate build directory to enable sustained-contact audio.
`-DMESHEDITOR_APP=OFF` omits the app executable; individual test targets build only their dependency closure.
