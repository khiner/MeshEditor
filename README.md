# MeshEditor

Real-time mesh viewer and editor, using Vulkan and ImGui.

![](screenshot.png)

For me to learn Vulkan, and to transition [mesh2audio](https://github.com/khiner/mesh2audio) to Vulkan with this project as the mesh library so it's just responsible for the audio/modeling side.

## Features

List of interesting features so far:
* Terse and direct usage of [Vulkan-Hpp](https://github.com/KhronosGroup/Vulkan-Hpp) with little indirection
* Change and recompile the SPIRV shader code at runtime
* Resource reflection: Use [`SPIRV-Cross`](https://github.com/KhronosGroup/SPIRV-Cross) to automatically create descriptor set layout bindings for all shader pipelines
* Instanced rendering of shared geometry with many model transforms
* Create/delete meshes and mesh instances
* Editable mesh primitives (Rect, Circle, Cube, IcoSphere, UVSphere, Torus, Cylinder, Cone)
* Camera rotate & zoom mouse/scrollwheel controls
* Simple camera + scene lighting model, visually matching Blender
* Flat/smooth/line rendering
* Hover-highlight vertices, edges, or faces
* Normal debugging: Render face/vertex normal lines
* Edge-detection-based silhouette outline of selected mesh/instance, embedded into the scene with accurate per-pixel depth
* Fast infinite grid with horizon fade
* Camera/lighting editing
* Object and view manipulation gizmos

## Build app

### Install dependencies

- Download and install the latest SDK from https://vulkan.lunarg.com/sdk/home
- Set the `VULKAN_SDK` environment variable.
  For example, add the following to your `.zshrc` file:
  ```shell
  export VULKAN_SDK="$HOME/VulkanSDK/{version}/macOS"
  ```

#### Mac

```shell
$ brew install cmake pkgconfig llvm
$ brew link llvm --force
```

#### Linux

(Only tested on Ubuntu.)

```shell
$ sudo apt install llvm libc++-dev libc++abi-dev
$ ln -s llvm-config-17 llvm-config
$ export PATH="$(llvm-config --bindir):$PATH"
```

Install GTK (for native file dialogs):

```shell
$ sudo apt install build-essential libgtk-3-dev
```

### Clone, clean, and build app

```shell
$ git clone --recurse-submodules git@github.com:khiner/MeshEditor.git
$ cd MeshEditor
$ mkdir build && cd make && cmake .. && make
```

## Stack

- [ImGui](https://github.com/ocornut/imgui) + [SDL3](https://github.comlibsdl-org/SDL) + [Vulkan](https://www.vulkan.org/): Immediate-mode UI/UX.
- [glm](https://github.com/g-truc/glm): Graphics math.
- [OpenMesh](https://gitlab.vci.rwth-aachen.de:9000/OpenMesh/OpenMesh): Main polyhedral mesh data structure.
- [entt](https://github.com/skypjack/entt): Entity Component System (ECS) for an efficient and scalable mixin-style architectural pattern
- [nativefiledialog-extended](https://github.com/btzynativefiledialog-extended): Native file dialogs.
- [ImGuizmo](https://github.com/CedricGuillemet/ImGuizmo): Mesh transform and camera rotation gizmos.
