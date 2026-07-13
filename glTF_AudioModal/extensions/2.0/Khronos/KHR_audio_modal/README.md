<!--
Copyright 2026 The Khronos Group Inc.
SPDX-License-Identifier: CC-BY-4.0
-->

# KHR\_audio\_modal

## Contributors <!-- omit in toc -->

- Karl Hiner, [@khiner](https://github.com/khiner)

Copyright 2026 The Khronos Group Inc. All Rights Reserved. glTF is a trademark of The Khronos Group Inc.
See [Appendix](#appendix-full-khronos-copyright-statement) for full Khronos Copyright Statement.

## Status <!-- omit in toc -->

Draft

## Dependencies <!-- omit in toc -->

Written against the glTF 2.0 spec.

## Table of Contents <!-- omit in toc -->

- [Overview](#overview)
- [Units](#units)
- [Defining Modal Models](#defining-modal-models)
  - [Modes](#modes)
  - [Sample Points](#sample-points)
  - [Accessor Requirements](#accessor-requirements)
  - [Mass Properties](#mass-properties)
- [Acoustic Materials](#acoustic-materials)
- [Attaching Models to Nodes](#attaching-models-to-nodes)
- [Audio Rendering](#audio-rendering)
  - [Excitation](#excitation)
  - [Synthesis](#synthesis)
  - [Acceleration Noise](#acceleration-noise)
- [Node Transforms and Scale](#node-transforms-and-scale)
- [Interaction with Other Extensions](#interaction-with-other-extensions)
- [Scope and Exclusions](#scope-and-exclusions)
- [Authoring Notes](#authoring-notes)
- [JSON Schema](#json-schema)
- [Object Model](#object-model)
- [Known Implementations](#known-implementations)
- [References](#references)
- [Appendix A: Deriving Modal Data](#appendix-a-deriving-modal-data)
- [Appendix: Full Khronos Copyright Statement](#appendix-full-khronos-copyright-statement)

## Overview

An impact on a near-rigid body excites damped vibrational modes, each radiating sound at a characteristic frequency and decay. Linear modal synthesis reproduces this by driving a bank of damped sinusoidal oscillators from contact forces, and is the established technique for physically based impact sound in interactive applications.

This extension allows a glTF asset to carry precomputed modal sound models. A modal model consists of, per mode, a frequency, a decay rate, and a spatial *mode shape*: a displacement vector field sampled at points on the object's surface, which determines how strongly a contact at a given position and direction excites the mode. Models are defined at the document level and instanced by nodes, and may reference an *acoustic material* describing the physical parameters they were derived from.

Modal analysis is performed at authoring time. This extension stores only its results. The data is independent of rendering technique and output sample rate: the same model may be rendered by a simple resonator bank or drive a full acoustic wave simulation. Contact forces are provided by the host application, typically from a physics simulation (see [Interaction with Other Extensions](#interaction-with-other-extensions)). Sound propagation, spatialization, and acoustic radiation transfer are out of scope (see [Scope and Exclusions](#scope-and-exclusions)).

This extension is purely additive: an implementation without audio support can ignore it, and assets SHOULD NOT list it in `extensionsRequired`.

## Units

This extension uses glTF core units (meters, radians, right-handed coordinates) and adds:

| Property | Units |
|-|-|
| `frequencies` accessor values | Hertz (Hz) |
| `decayRates` accessor values | Per second (s⁻¹) |
| `positions` accessor values | Node-local space (meters after the global transform, as mesh `POSITION`) |
| `shapes` accessor values | Per square-root kilogram (kg⁻¹ᐟ²) |
| `material.density` | Kilogram per cubic meter (kg·m⁻³) |
| `material.youngsModulus` | Pascal (Pa) |
| `material.alpha` | Per second (s⁻¹) |
| `material.beta` | Second (s) |
| Excitation impulse | Newton second (N·s) |

## Defining Modal Models

Modal models and acoustic materials are arrays in a `KHR_audio_modal` object on the glTF root, referenced by index.

```json
{
    "extensionsUsed": ["KHR_audio_modal"],
    "extensions": {
        "KHR_audio_modal": {
            "materials": [
                { "name": "Ceramic", "density": 2700, "youngsModulus": 7.2e10,
                  "poissonRatio": 0.19, "alpha": 6, "beta": 1e-7 }
            ],
            "models": [
                {
                    "name": "Bowl",
                    "frequencies": 10,
                    "decayRates": 11,
                    "positions": 12,
                    "shapes": 13,
                    "material": 0
                }
            ]
        }
    }
}
```

| | Type | Description | Required |
|-|-|-|-|
| **models** | modal model `[1-*]` | An array of modal sound models. | :white_check_mark: Yes |
| **materials** | acoustic material `[1-*]` | An array of acoustic materials. | No |

Each modal model has the following properties:

| | Type | Description | Required |
|-|-|-|-|
| **frequencies** | `integer` | Accessor of per-mode frequencies, in Hz. | :white_check_mark: Yes |
| **decayRates** | `integer` | Accessor of per-mode amplitude decay rates *d*, in s⁻¹. A mode's amplitude envelope is *e*⁻ᵈᵗ. | :white_check_mark: Yes |
| **positions** | `integer` | Accessor of sample point positions, in the node's local space. | :white_check_mark: Yes |
| **shapes** | `integer` | Accessor of per-mode, per-sample-point displacement vectors. | :white_check_mark: Yes |
| **indices** | `integer` | Accessor of triangle indices into the sample points, defining an interpolation surface. | No |
| **material** | `integer` | The index of the acoustic material the model was derived from. | No |
| **massProperties** | `object` | The object's mass, center of mass, and inertia, used for [acceleration noise](#acceleration-noise) and mass-based scaling. | No |

### Modes

Each mode *n* is a damped sinusoid. `frequencies` values are the *observed* (damped) frequencies and MUST be positive. `decayRates` values MUST be non-negative. Modes representing rigid-body motion (zero frequency) MUST NOT be included.

Modes SHOULD be ordered by decreasing perceptual importance, so that implementations rendering only the first *N* modes degrade gracefully.

### Sample Points

Mode shapes are sampled at a set of points on the object's surface, defined by `positions` in the node's local space. Sample points are independent of any mesh: they need not coincide with render mesh vertices, and a model may be attached to a node without a mesh.

The mode shape **φ**ₙ(**p**) at an arbitrary surface position **p** is the `shapes` value of the sample point nearest to **p**. When `indices` is present, implementations SHOULD instead project **p** to the nearest point on the triangulated sample surface and interpolate the containing triangle's three `shapes` values barycentrically at that point.

Mode shapes SHOULD be mass-normalized displacement eigenvectors (**Φ**ᵀ**M** **Φ** = **I** with the mass matrix in kilograms, see [Appendix A](#appendix-a-deriving-modal-data)), making relative loudness across models and strike strengths physically meaningful. Models with any other normalization (e.g. fitted from recordings) MUST be expressed in the same form, so that the [excitation](#excitation) and [synthesis](#synthesis) definitions below produce the intended relative amplitudes.

### Accessor Requirements

With *M* modes and *P* sample points:

| Property | Accessor Type | Component Type | Count |
|-|-|-|-|
| `frequencies` | `"SCALAR"` | `5126` (FLOAT) | *M* |
| `decayRates` | `"SCALAR"` | `5126` (FLOAT) | *M* |
| `positions` | `"VEC3"` | `5126` (FLOAT) | *P* |
| `shapes` | `"VEC3"` | `5126` (FLOAT) | *M* × *P* |
| `indices` | `"SCALAR"` | `5125` (UNSIGNED_INT) or `5123` (UNSIGNED_SHORT) | multiple of 3 |

*M* and *P* MUST each be at least 1. `shapes` is mode-major: element *m*·*P* + *i* is the shape of mode *m* at sample point *i*. Rendering only the first *N* modes therefore reads a prefix of each accessor. `indices` values MUST be less than *P*. All accessor values MUST be finite (no `NaN` or infinity).

### Mass Properties

`massProperties` records the object's rigid-body mass distribution, at the model's reference size ([Node Transforms and Scale](#node-transforms-and-scale)). It is optional, and is consumed by [acceleration noise](#acceleration-noise) and by mass-based adjustments. Its fields mirror `KHR_physics_rigid_bodies`' rigid-body motion, so the same physical body can be described in either extension.

| | Type | Description | Required |
|-|-|-|-|
| **mass** | `number` | Mass in kg. | :white_check_mark: Yes |
| **centerOfMass** | `number[3]` | Center of mass in the node's local space. Default `[0,0,0]`. | No |
| **inertiaDiagonal** | `number[3]` | Principal moments of inertia, in kg·m². Default `[0,0,0]` (a point mass). | No |
| **inertiaOrientation** | `number[4]` | Unit quaternion rotating the principal inertia axes into local space. Default `[0,0,0,1]`. | No |

When the node also has a `KHR_physics_rigid_bodies` dynamic rigid body, that body's mass properties are authoritative for contact dynamics, and this extension's `massProperties`, if present, MUST be ignored. The two describe one physical body, and grounding the sound in the mass the simulation integrates keeps the audible impact consistent with the visible motion. Authored values SHOULD agree. A kinematic or static rigid body carries no finite mass for the object, so this extension's `massProperties` apply in that case.

When a model omits `massProperties` and no rigid body supplies them, an implementation MAY compute the same quantities from the node's mesh and ρ when the material specifies ρ and the mesh is watertight.

## Acoustic Materials

An acoustic material records the physical parameters a model was derived from. It is not required for rendering, but enables physically based adjustments: recomputing decay rates, rescaling models under uniform scale ([Node Transforms and Scale](#node-transforms-and-scale)), and estimating contact durations.

| | Type | Description | Required |
|-|-|-|-|
| **density** | `number` | Mass density ρ, in kg·m⁻³. | No |
| **youngsModulus** | `number` | Young's modulus *E*, in Pa. | No |
| **poissonRatio** | `number` | Poisson's ratio ν, in (−1, 0.5). | No |
| **alpha** | `number` | Rayleigh damping coefficient α (mass-proportional), in s⁻¹. | No |
| **beta** | `number` | Rayleigh damping coefficient β (stiffness-proportional), in s. | No |

`alpha` and `beta` relate decay rate to frequency by *d* = (α + βω²)/2, where ω is the mode's undamped angular frequency ([Appendix A](#appendix-a-deriving-modal-data), which also lists representative values for common materials).

## Attaching Models to Nodes

A node instances a modal model with a `KHR_audio_modal` extension object:

```json
"nodes": [
    {
        "mesh": 0,
        "extensions": {
            "KHR_audio_modal": { "model": 0, "gain": 1.0 }
        }
    }
]
```

| | Type | Description | Required |
|-|-|-|-|
| **model** | `integer` | The index of the modal model instanced by this node. | :white_check_mark: Yes |
| **gain** | `number` | Linear amplitude scale applied to this instance's output. | No, default: `1.0` |

Multiple nodes MAY reference the same model. Each instance MUST have independent oscillator state. A node using `EXT_mesh_gpu_instancing` instances the model once per render instance ([Interaction with Other Extensions](#interaction-with-other-extensions)).

## Audio Rendering

A conformant audio renderer synthesizes each model instance's response to contact events as follows. This defines a source *signal* only. How that signal radiates into the scene (directivity, distance, occlusion, reverberation) is out of scope: an implementation MAY layer any radiation or propagation model over it, and absent one SHOULD treat it as a monophonic source at the node's origin.

### Excitation

An excitation is an impulse **j** (in N·s) applied at a surface position **p** at time *t*₀. Both MUST be expressed in the node's local space: **p** by the inverse of the node's global transform, and **j** by the inverse of the global transform's rotation only, preserving the impulse's physical magnitude.

The excitation amplitude of mode *n* is the projection of the impulse onto the mode shape at the contact position:

*a*ₙ = **φ**ₙ(**p**) · **j**

A contact applies force over a finite duration τ, the Hertz contact time, longer for softer and heavier contacts. Each mode is excited by that force's spectrum at its frequency, so implementations SHOULD scale each *a*ₙ by *F̂*(*f*ₙ), the force-pulse spectrum normalized to *F̂*(0) = 1. A half-sine pulse of duration τ is the recommended default: *F̂* is near unity well below 1/(2τ), reaches −3 dB near 1/(2τ), and about −10 dB at 1/τ, so a mode well above 1/τ is barely excited. A resonator bank MAY realize this by driving each mode with the sampled pulse in place of an ideal impulse, which produces the same per-mode scaling without a separate filtering step. Sustained contacts MAY be rendered as a sequence of such excitations.

### Synthesis

The instance's output signal is the superposition of its modes' responses to all excitations:

*s*(*t*) = *g* · Σₖ Σₙ *a*ₙₖ · *e*^(−*d*ₙ(*t*−*t*ₖ)) · sin(2π*f*ₙ(*t*−*t*ₖ)),  summed over excitations *k* with *t*ₖ ≤ *t*

where *g* is the instance `gain` and *a*ₙₖ is the excitation amplitude of mode *n* for the excitation at time *t*ₖ. Equivalently, implementations MAY use any resonator with matching impulse response (e.g. two-pole IIR filters), which also admits arbitrary excitation signals.

- Relative amplitudes among modes, excitations, and model instances are normative. The absolute output level is implementation-defined, applied uniformly to all instances.
- Excitations MUST superpose linearly.
- Implementations MUST NOT produce aliased output from modes with *f*ₙ at or above the output Nyquist frequency. Equivalently, such modes contribute no output.
- Implementations MAY render only a subset of modes (e.g. the first *N*, or a psychoacoustically culled set) to meet performance constraints.

Model parameters are continuous-time. Synthesis at any output sample rate MUST NOT change pitch or decay.

### Acceleration Noise

Besides ringing its modes, a struck body recoils as a whole: the contact accelerates its rigid-body degrees of freedom, radiating a short broadband transient (the contact "click"). It is distinct from the modal response and is the dominant sound for small, stiff bodies whose modes lie above the audible range. Implementations SHOULD render it when the object's [mass properties](#mass-properties) are available.

The same excitation (impulse **j** at position **p**, contact time τ) drives it. With mass *M*, center of mass **c**, and inertia **I** (all in the node's local space), the contact imparts a linear and angular velocity change

Δ**v** = **j** / *M*,  Δ**ω** = **I**⁻¹ ((**p** − **c**) × **j**)

delivered over the contact through the same finite force pulse used for the excitation (∫ **F** d*t* = **j**), so the body's rigid acceleration follows that pulse's shape. A compact body recoiling without changing volume radiates as an acoustic dipole, whose pressure is proportional to the *time-derivative* of that acceleration and to the body's displaced volume. The source signal therefore has that derivative shape, broadband up to ≈1/τ, so shorter contacts click brighter. A resonator bank MAY produce it directly as the derivative of the contact force pulse, scaled per strike, and superpose it on the modal output ([Synthesis](#synthesis)).

Its relative shape and scaling with contact strength are normative. Its absolute level, like the modal output's, is implementation-defined. Its dipole directivity, like all radiation here, is out of scope: rendered omnidirectionally by default (the same approximation the modal core makes), or shaped by whatever radiation model an implementation layers on ([Scope and Exclusions](#scope-and-exclusions)).

## Node Transforms and Scale

A modal model describes its object at a fixed physical size: the node's global transform in the scene's initial state (before any animation). Sample point positions transform with the node like mesh vertices.

Vibration frequencies are not invariant under scaling. If the node's global scale is later changed *uniformly* by a factor γ relative to the initial state, implementations SHOULD adjust the model: with each mode's undamped angular frequency ωₙ = √((2π*f*ₙ)² + *d*ₙ²), scale ωₙ → ωₙ/γ and **φ**ₙ → γ⁻³ᐟ² **φ**ₙ, recompute *d*ₙ = (α + βωₙ²)/2 from the scaled ωₙ when the model has a material with `alpha` and `beta` (otherwise leave *d*ₙ unchanged), and derive *f*ₙ = √(ωₙ² − *d*ₙ²)/2π. Implementations that do not support rescaling SHOULD render the model unmodified.

Behavior is undefined whenever the node's global transform contains non-uniform scale: non-uniform scaling changes the object's mode shapes and frequencies in ways that cannot be recovered from precomputed data (and leaves the transform's rotation ill-defined). Authors requiring a non-uniformly scaled object bake the scale into the geometry and analyze the result.

## Interaction with Other Extensions

**KHR_physics_rigid_bodies**: Although excitations may come from any host source, the typical source is a rigid-body simulation, where a collision yields the two required quantities: a contact impulse and a world-space contact position, as in that extension's `rigid_body/applyPointImpulse` interactivity node. Contacts on a collider SHOULD excite the modal model of the collider node or its nearest ancestor that has one. The mechanism by which a simulation reports contact events is beyond the scope of this specification.

**EXT_mesh_gpu_instancing**: A node using GPU instancing instantiates its modal model once per render instance, each with independent oscillator state and the node's `gain`. An excitation targets a single render instance. Attribution is host logic, like contact reporting. Excitation mapping ([Excitation](#excitation)) uses the composed instance transform (the transform applied to the instance's vertices for rendering, as defined by `EXT_mesh_gpu_instancing`) in place of the node's global transform, and each instance's output is a monophonic source at that instance's origin. The model describes the object under identity instance transform. An instance's uniform scale relative to that reference (its `SCALE` attribute, composed with any node scale change) is a uniform scale change under [Node Transforms and Scale](#node-transforms-and-scale), and behavior is undefined when the composed transform contains non-uniform scale.

**KHR_audio_emitter / KHR_audio_graph** (proposals): This extension generates source signals and does not define emitters, listeners, or spatialization. When a node with a modal model also has an audio emitter, the synthesized signal SHOULD be routed to that emitter as a source.

An acoustic material is distinct from a physics material (friction, restitution) and a render material. A node may carry all three.

## Scope and Exclusions

The following are deliberately out of scope. Each composes *with* the modal core rather than replacing it, and may be layered by future extensions:

- **Acoustic radiation transfer** (listener-position-dependent, per-mode amplitude fields, e.g. FFAT maps or multipole expansions). Without it, a model radiates omnidirectionally, including [acceleration noise](#acceleration-noise) directivity.
- **Sound propagation and spatialization**: distance attenuation, occlusion, reverberation, and listener modeling belong to audio-emitter and platform layers.
- **Contact-event plumbing**: how a physics engine reports impulses to the audio system is application logic.
- **Sustained-contact excitation synthesis**: surface roughness profiles and scrape/roll force models. This extension defines only how a given excitation drives the modes.
- **Nonlinear vibration**: mode coupling in thin shells (cymbals, sheet metal), fracture, and contact-dependent damping.
- **Recorded-sample hybrids**: mixing recorded impact audio with the synthesized modes for detail beyond linear modal synthesis.
- **Modal analysis itself**: meshing, FEM, and eigensolves happen at authoring time ([Appendix A](#appendix-a-deriving-modal-data)).

## Authoring Notes

*This section is non-normative.*

Linear modal models are valid for near-rigid solid bodies under small deformations. Thin shells and strongly nonlinear objects are poorly reproduced. Deriving modes with FEM requires a watertight, tet-meshable solid at authoring time, but this extension imposes no geometry requirements at runtime. Sample points are self-contained, so render meshes may be remeshed, decimated, or LOD-swapped without invalidating the audio data.

Modes are typically band-limited to [20 Hz, 20 kHz] at authoring time. A few tens of modes are perceptually sufficient for most objects, and a few hundred for large or broadband ones. Sample point density is an authoring choice: a handful of points suffices for coarse position-dependent timbre, while per-vertex density captures fine spatial variation at proportional cost.

## JSON Schema

- [glTF.KHR_audio_modal.schema.json](schema/glTF.KHR_audio_modal.schema.json)
- [glTF.KHR_audio_modal.model.schema.json](schema/glTF.KHR_audio_modal.model.schema.json)
- [glTF.KHR_audio_modal.material.schema.json](schema/glTF.KHR_audio_modal.material.schema.json)
- [node.KHR_audio_modal.schema.json](schema/node.KHR_audio_modal.schema.json)

## Object Model

The following JSON pointer is defined for use with the glTF Object Model (e.g. by `KHR_animation_pointer` and `KHR_interactivity`):

| Pointer | Type |
|-|-|
| `/nodes/{}/extensions/KHR_audio_modal/gain` | `float` |

Modal model and acoustic material data are static and not addressable.

## Known Implementations

- [MeshEditor](https://github.com/khiner/MeshEditor) — authoring (FEM modal analysis from meshes) and rendering (coupled-form resonator bank), with extension import/export in progress.

## References

- K. L. Johnson. *Contact Mechanics.* Cambridge University Press, 1985.
- K. van den Doel, D. K. Pai. *The Sounds of Physical Shapes.* Presence 7(4), 1998.
- K. van den Doel, P. G. Kry, D. K. Pai. *FoleyAutomatic: Physically-based Sound Effects for Interactive Simulation and Animation.* SIGGRAPH 2001.
- J. F. O'Brien, C. Shen, C. M. Gatchalian. *Synthesizing Sounds from Rigid-Body Simulations.* SCA 2002.
- D. L. James, J. Barbič, D. K. Pai. *Precomputed Acoustic Transfer: Output-sensitive, Accurate Sound Generation for Geometrically Complex Vibration Sources.* SIGGRAPH 2006.
- C. Zheng, D. L. James. *Rigid-Body Fracture Sound with Precomputed Soundbanks.* SIGGRAPH 2010.
- C. Zheng, D. L. James. *Toward High-Quality Modal Contact Sound.* SIGGRAPH 2011.
- J. N. Chadwick, C. Zheng, D. L. James. *Precomputed Acceleration Noise for Improved Rigid-Body Sound.* SIGGRAPH 2012.
- T. R. Langlois, S. S. An, K. K. Jin, D. L. James. *Eigenmode Compression for Modal Sound Models.* SIGGRAPH 2014.
- J.-H. Wang, D. L. James. *KleinPAT: Optimal Mode Conflation for Time-Domain Precomputation of Acoustic Transfer.* SIGGRAPH 2019.
- S. Clarke et al. *RealImpact: A Dataset of Impact Sound Fields for Real Objects.* CVPR 2023.
- D. Menzies. *Physically Motivated Environmental Sound Synthesis for Virtual Worlds.* EURASIP JASMP, 2010.

## Appendix A: Deriving Modal Data

*This appendix is non-normative.*

The standard authoring pipeline discretizes the object as a tetrahedral finite-element mesh, assembles mass and stiffness matrices **M**, **K** from the material's ρ, *E*, ν, and solves the generalized eigenproblem

**K** **Φ** = **M** **Φ** **Λ**,  **Φ**ᵀ**M** **Φ** = **I**

The six zero eigenvalues of an unconstrained body (rigid translations and rotations) are discarded. For each remaining eigenpair (λₙ, **φ**ₙ), with Rayleigh damping **C** = α**M** + β**K**:

- Undamped angular frequency: ωₙ = √λₙ
- Decay rate: *d*ₙ = (α + βωₙ²)/2
- Observed frequency: *f*ₙ = √(ωₙ² − *d*ₙ²) / 2π
- Stored shape at a sample point: the 3-vector block of **φ**ₙ at the nearest surface vertex

Conversions: T60 = ln(1000)/*d* ≈ 6.908/*d*; quality factor Q = π*f*/*d*. Under uniform geometric scaling by γ, ω → ω/γ and mass-normalized **φ** → γ⁻³ᐟ² **φ** (Zheng & James 2010, Appendix E).

Representative material parameters (Wang & James 2019, Table 4; SI units):

| Material | ρ (kg/m³) | E (Pa) | ν | α (s⁻¹) | β (s) |
|-|-|-|-|-|-|
| Ceramic | 2700 | 7.2e10 | 0.19 | 6 | 1e-7 |
| Glass | 2600 | 6.2e10 | 0.20 | 1 | 1e-7 |
| Wood | 750 | 1.1e10 | 0.25 | 60 | 2e-6 |
| Plastic | 1070 | 1.4e9 | 0.35 | 30 | 1e-6 |
| Iron | 8000 | 2.1e11 | 0.28 | 5 | 1e-7 |
| Polycarbonate | 1190 | 2.4e9 | 0.37 | 0.5 | 4e-7 |
| Steel | 7850 | 2.0e11 | 0.29 | 5 | 3e-8 |

Models may also be fitted from measured impact recordings (e.g. spectral peak picking for frequencies, log-envelope regression for decay rates, per-strike-point amplitudes for shapes), in which case decay rates are unconstrained by the Rayleigh model and radiation effects may be baked into the shapes.

## Appendix: Full Khronos Copyright Statement

Copyright 2026 The Khronos Group Inc.

This specification is protected by copyright laws and contains material proprietary
to Khronos. Except as described by these terms, it or any components
may not be reproduced, republished, distributed, transmitted, displayed, broadcast,
or otherwise exploited in any manner without the express prior written permission
of Khronos.

This specification has been created under the Khronos Intellectual Property Rights
Policy, which is Attachment A of the Khronos Group Membership Agreement available at
https://www.khronos.org/files/member_agreement.pdf. Khronos grants a conditional
copyright license to use and reproduce the unmodified specification for any purpose,
without fee or royalty, EXCEPT no licenses to any patent, trademark or other
intellectual property rights are granted under these terms. Parties desiring to
implement the specification and make use of Khronos trademarks in relation to that
implementation, and receive reciprocal patent license protection under the Khronos
IP Policy must become Adopters under the process defined by Khronos for this specification;
see https://www.khronos.org/conformance/adopters/file-format-adopter-program.

Some parts of this Specification are purely informative and do not define requirements
necessary for compliance and so are outside the Scope of this Specification. These
parts of the Specification are marked as being non-normative, or identified as
**Implementation Notes**.

Where this Specification includes normative references to external documents, only the
specifically identified sections and functionality of those external documents are in
Scope. Requirements defined by external documents not created by Khronos may contain
contributions from non-members of Khronos not covered by the Khronos Intellectual
Property Rights Policy.

Khronos makes no, and expressly disclaims any, representations or warranties,
express or implied, regarding this specification, including, without limitation:
merchantability, fitness for a particular purpose, non-infringement of any
intellectual property, correctness, accuracy, completeness, timeliness, and
reliability. Under no circumstances will Khronos, or any of its Promoters,
Contributors or Members, or their respective partners, officers, directors,
employees, agents or representatives be liable for any damages, whether direct,
indirect, special or consequential damages for lost revenues, lost profits, or
otherwise, arising from or in connection with these materials.

Khronos® and Vulkan® are registered trademarks, and ANARI™, WebGL™, glTF™, NNEF™, OpenVX™,
SPIR™, SPIR&#8209;V™, SYCL™, OpenVG™ and 3D Commerce™ are trademarks of The Khronos Group Inc.
OpenXR™ is a trademark owned by The Khronos Group Inc. and is registered as a trademark in
China, the European Union, Japan and the United Kingdom. OpenCL™ is a trademark of Apple Inc.
and OpenGL® is a registered trademark and the OpenGL ES™ and OpenGL SC™ logos are trademarks
of Hewlett Packard Enterprise used under license by Khronos. ASTC is a trademark of
ARM Holdings PLC. All other product names, trademarks, and/or company names are used solely
for identification and belong to their respective owners.
