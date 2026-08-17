# glTF Physical Audio

This repository describes glTF extensions for physically based contact sound. It allows glTF scene authors to attach precomputed modal sound models (per-mode frequencies, decay rates, and spatial mode shapes), acoustic material properties, and acoustic surface finishes to scene nodes, so that audio renderers can synthesize physically plausible sound for rigid bodies responding to impacts and sustained contact.

It is the acoustic counterpart to [glTF_Physics](https://github.com/eoineoineoin/glTF_Physics), over the same bodies and the same contact events: where that solves the motion of a rigid body, this solves the sound of small vibrations about that motion.

## KHR_audio_rigid_bodies

Defines document-level modal sound models, acoustic materials, and acoustic surfaces, instanced by nodes. Specifies how contact impulses and sustained contacts excite a model's modes, and the normative synthesized response.

Impact excitation, synthesis, acceleration noise, and radiation are settled, and an implementation supporting them alone is conformant. Sustained contact is under active development and is gathered in one section of its own.

[Specification](extensions/2.0/Khronos/KHR_audio_rigid_bodies/README.md)

## Design Decisions

[Rationale](DesignDecisions.md) for what was decided, what was rejected, and what would prompt revisiting. Not part of the specification.

## Known Implementations

- [MeshEditor](https://github.com/khiner/MeshEditor) — authoring (FEM modal analysis from meshes) and rendering (coupled-form resonator bank), with extension import/export in progress.
