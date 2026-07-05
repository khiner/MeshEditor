# glTF Modal Audio

This repository describes a glTF extension for physically based impact sound. It allows glTF scene authors to attach precomputed modal sound models (per-mode frequencies, decay rates, and spatial mode shapes) and acoustic material properties to scene nodes, so that audio renderers can synthesize physically plausible sound for rigid bodies responding to contact forces.

## KHR_audio_modal

Defines document-level modal sound models and acoustic materials, instanced by nodes. Specifies how contact impulses excite a model's modes and the normative synthesized response.

[Specification](extensions/2.0/Khronos/KHR_audio_modal/README.md)

## Known Implementations

- [MeshEditor](https://github.com/khiner/MeshEditor) — authoring (FEM modal analysis from meshes) and rendering (Faust-based resonator bank), with extension import/export in progress.
