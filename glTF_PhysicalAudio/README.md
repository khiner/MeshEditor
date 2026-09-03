# glTF Physical Audio

This repository defines glTF extensions for physically based contact sound. Scene authors can attach modal sound models, acoustic materials, and surface finishes to nodes for impact and sustained-contact synthesis.

It complements [glTF_Physics](https://github.com/eoineoineoin/glTF_Physics) by generating sound from the same rigid bodies and contact events.

## KHR_audio_rigid_bodies

Defines document-level modal models, acoustic materials, and acoustic surfaces instantiated by nodes. It specifies modal excitation and the normative synthesized response.

Impact excitation, synthesis, acceleration noise, and radiation form the conformant core. Sustained contact remains under active development in a dedicated section.

[Specification](extensions/2.0/Khronos/KHR_audio_rigid_bodies/README.md)

## Design Decisions

[Rationale](DesignDecisions.md) records decisions, rejected alternatives, and revision criteria. It is not part of the specification.

## Known Implementations

- [MeshEditor](https://github.com/khiner/MeshEditor) — FEM modal authoring and coupled-form resonator rendering; extension import and export are in progress.
