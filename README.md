# mesh2audio

Import or generate 3D models and transform them into real-time playable physical audio models!

The generated audio model can be played in real-time by "striking" (clicking) on mesh vertices in the 3D mesh viewer or by using an audio input device (such as a microphone) to excite the vertices.

- Project video: https://youtu.be/RwxgOHVBDvc
- Example audio output is in the [sound_samples](sound_samples/) directory.

This project started as the final project for Karl Hiner and Ben Wilfong for GA Tech CSE-6730 Modeling & Simulation, Spring 2023.
_The code as it was at the end of the school project is [here](https://github.com/GATech-CSE-6730-Spring-2023-Project/mesh2audio)._
_This original repo also supports low-fidelity axissymmetric audio model generation._

## Build app

### Install dependencies

#### Mac

```shell
$ brew install glew llvm eigen
```

#### Linux

(Only tested on Ubuntu.)

```shell
$ sudo apt install llvm libc++-dev libc++abi-dev libeigen3-dev
$ ln -s llvm-config-16 llvm-config
$ export PATH="$(llvm-config --bindir):$PATH"
```

Install GTK (for native file dialogs) and OpenGL dependencies:

```shell
$ sudo apt install build-essential libgtk-3-dev libglu1-mesa-dev freeglut3-dev mesa-common-dev libglew-dev
```

### Clone, clean, and build app

```shell
$ git clone --recurse-submodules git@github.com:khiner/mesh2audio.git
$ cd mesh2audio
```

- **Clean:**
  - Clean up everything: `./script/Clean`
  - Clean debug build only: `./script/Clean -d [--debug]`
  - Clean release build only: `./script/Clean -r [--release]`
- **Build:**
  - Debug build (default): `./script/Build`
  - Release build: `./script/Build -r [--release]`

Debug build is generated in the `./build` directory relative to project (repo) root.
Release build is generated in `./build-release`.

To run the freshly built application:

```sh
# The application assumes it's being run from the build directory when locating its resource files.
$ cd build # or build-release
$ ./mesh2audio
```

## Stack

- [ImGui](https://github.com/ocornut/imgui) + [SDL3](https://github.comlibsdl-org/SDL): Immediate-mode UI/UX.
- [Faust](https://github.com/grame-cncm/faust): Render the mesh to an audiograph, with real-time interactive vertex excitation.
- [miniaudio](https://github.com/mackron/miniaudio): Continuously render themodal physical model of the input 3D volumetric mesh to audio.
- [glm](https://github.com/g-truc/glm): Graphics math.
- [OpenMesh](https://gitlab.vci.rwth-aachen.de:9000/OpenMesh/OpenMesh): Mainpolyhedral mesh representation data structure.
- [tetgen](https://github.com/libigl/tetgen): Convert triangular 3D surfacemeshes into tetrahedral meshes.
- 3D FEM: [VegaFEM](https://github.com/grame-cncm/faust/tree/master-dev/tools/physicalModeling/mesh2faust/vega) for generating mass/stiffness matrices + [Spectra](https://github.com/yixuan/spectra) for finding eigenvalues/vectors.
- [ReactPhysics3D](https://github.com/DanielChappuis/reactphysics3d/treedevelop): Collision detection and physics.
- [nativefiledialog-extended](https://github.com/btzynativefiledialog-extended): Native file dialogs.
- [nanosvg](https://github.com/memononen/nanosvg): Read path vertices fromSVG files.
- [ImPlot](https://github.com/epezent/implot): Plotting.
- [ImGuizmo](https://github.com/CedricGuillemet/ImGuizmo): Mesh transform andcamera rotation gizmos.
- [ImSpinner](https://github.com/dalerank/imspinner): Wicked cool loadingspinners for ImGui.
- [libnpy](https://github.com/llohse/libnpy): Read `.npy` files (used in WIP [RealImpact](https://github.com/samuel-clarke/RealImpact) explorer).
