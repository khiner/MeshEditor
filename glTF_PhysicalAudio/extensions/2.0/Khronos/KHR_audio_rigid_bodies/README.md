<!--
Copyright 2026 The Khronos Group Inc.
SPDX-License-Identifier: CC-BY-4.0
-->

# KHR\_audio\_rigid\_bodies

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
- [Acoustic Surfaces](#acoustic-surfaces)
  - [Mesoscale Structure](#mesoscale-structure)
  - [Surface Profiles](#surface-profiles)
- [Attaching to Nodes](#attaching-to-nodes)
- [Audio Rendering](#audio-rendering)
  - [Excitation](#excitation)
  - [Synthesis](#synthesis)
  - [Acceleration Noise](#acceleration-noise)
  - [Sustained Contact](#sustained-contact)
    - [Contact State](#contact-state)
    - [Composite Surface](#composite-surface)
    - [Contact Force](#contact-force)
    - [Exciting the Modes](#exciting-the-modes)
    - [Vibrational Coupling](#vibrational-coupling)
- [Node Transforms and Scale](#node-transforms-and-scale)
- [Interaction with Other Extensions](#interaction-with-other-extensions)
- [Scope and Exclusions](#scope-and-exclusions)
- [Authoring Notes](#authoring-notes)
- [JSON Schema](#json-schema)
- [Object Model](#object-model)
- [Known Implementations](#known-implementations)
- [References](#references)
- [Appendix A: Deriving Modal Data](#appendix-a-deriving-modal-data)
- [Appendix B: Reference Contact Force Model](#appendix-b-reference-contact-force-model)
- [Appendix: Full Khronos Copyright Statement](#appendix-full-khronos-copyright-statement)

## Overview

A near-rigid body in contact makes sound. An impact excites its damped vibrational modes, each radiating at a characteristic frequency and decay. A contact that persists drives those same modes continuously as the irregularities of the two surfaces ride over one another, the mechanism known as *roughness excitation*, which covers sliding, scraping, and rolling alike. Linear modal synthesis reproduces both by driving a bank of damped sinusoidal oscillators from contact forces, and is the established technique for physically based contact sound in interactive applications.

This extension is the acoustic counterpart to `KHR_physics_rigid_bodies`, over the same bodies and the same contact events. Where that extension solves the motion of a rigid body, this one solves the sound of small vibrations about that motion. It allows a glTF asset to carry precomputed modal sound models, the acoustic materials they were derived from, and the acoustic surfaces that govern sustained contact. Models, materials, and surfaces are defined at the document level and instanced by nodes.

A modal model consists of, per mode, a frequency, a decay rate, and a spatial *mode shape*: a displacement vector field sampled at points on the object's surface, which determines how strongly a contact at a given position and direction excites the mode. Modal analysis is performed at authoring time. This extension stores only its results. The data is independent of rendering technique and output sample rate: the same model may be rendered by a simple resonator bank or drive a full acoustic wave simulation. Contact state is provided by the host application, typically from a physics simulation (see [Interaction with Other Extensions](#interaction-with-other-extensions)). Sound propagation, spatialization, and acoustic radiation transfer are out of scope (see [Scope and Exclusions](#scope-and-exclusions)).

This extension is purely additive: an implementation without audio support can ignore it, and assets SHOULD NOT list it in `extensionsRequired`. Every rendering feature it defines is optional beyond a small set of requirements that forbid audible failure modes rather than mandate an algorithm.

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
| `surface.roughness`, `surface.correlationLength`, `surface.sampleSpacing` | Meter (m) |
| `surface.profile` accessor values | Meter (m) |
| `surface.spectralSlope` | Dimensionless |
| Excitation impulse | Newton second (N·s) |
| Contact force | Newton (N) |
| Slip and sweep velocity | Meter per second (m·s⁻¹) |
| Friction coefficient | Dimensionless |

## Defining Modal Models

Modal models, acoustic materials, and acoustic surfaces are arrays in a `KHR_audio_rigid_bodies` object on the glTF root, referenced by index.

```json
{
    "extensionsUsed": ["KHR_audio_rigid_bodies"],
    "extensions": {
        "KHR_audio_rigid_bodies": {
            "acousticMaterials": [
                { "name": "Ceramic", "density": 2700, "youngsModulus": 7.2e10,
                  "poissonRatio": 0.19, "alpha": 6, "beta": 1e-7 },
                { "name": "Stone", "density": 2500, "youngsModulus": 5.0e10,
                  "poissonRatio": 0.25 }
            ],
            "acousticSurfaces": [
                { "name": "Glazed", "roughness": 3e-7, "correlationLength": 2e-5,
                  "spectralSlope": -1.7, "material": 0 },
                { "name": "Tiled floor", "roughness": 8e-6, "correlationLength": 8e-5,
                  "normalTexture": { "index": 2, "texCoord": 0 }, "material": 1 }
            ],
            "modalModels": [
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
| **modalModels** | modal model `[1-*]` | An array of modal sound models. | No |
| **acousticMaterials** | acoustic material `[1-*]` | An array of acoustic materials. | No |
| **acousticSurfaces** | acoustic surface `[1-*]` | An array of acoustic surfaces. | No |

All three arrays are optional. An asset may carry surfaces without models, which describes bodies that color the sound of contacts against them without sounding themselves.

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

An acoustic material records the bulk physical parameters of a solid. It is not required for rendering, but enables physically based adjustments: recomputing decay rates, rescaling models under uniform scale ([Node Transforms and Scale](#node-transforms-and-scale)), and deriving contact stiffness and duration.

Materials serve two roles. A modal model references one to record the parameters it was derived from. An acoustic surface references one to supply the elastic constants that set the stiffness of a contact against it, which is why a silent body such as a floor may carry a material with no model. `alpha` and `beta` are consumed only in the first role.

| | Type | Description | Required |
|-|-|-|-|
| **density** | `number` | Mass density ρ, in kg·m⁻³. | No |
| **youngsModulus** | `number` | Young's modulus *E*, in Pa. | No |
| **poissonRatio** | `number` | Poisson's ratio ν, in (−1, 0.5). | No |
| **alpha** | `number` | Rayleigh damping coefficient α (mass-proportional), in s⁻¹. | No |
| **beta** | `number` | Rayleigh damping coefficient β (stiffness-proportional), in s. | No |

`alpha` and `beta` relate decay rate to frequency by *d* = (α + βω²)/2, where ω is the mode's undamped angular frequency ([Appendix A](#appendix-a-deriving-modal-data), which also lists representative values for common materials).

## Acoustic Surfaces

An acoustic surface records the finish of a body's surface below the scale of its collision geometry: the structure that two bodies in sustained contact ride over, which is the source of scraping and rolling sound. It is a *surface* property, independent of the bulk material, so a polished and a sandblasted steel object share a material and differ here.

A surface is described at two scales below the collision geometry, following the levels of Ren et al. 2010:

- **Mesoscale**, the visible bumpiness of the surface: tiling, corrugation, knurling, grain. Authored as a normal map, and large enough to see.
- **Microscale**, the asperities of the finish itself: too fine to see and too fine for the physics solver, described statistically or by a measured track.

The collision geometry supplies the third and coarsest level. Between them the two scales here cover everything the geometry does not.

| | Type | Description | Required |
|-|-|-|-|
| **roughness** | `number` | Root-mean-square asperity height σ, in meters. | No, default: `2e-6` |
| **correlationLength** | `number` | Lateral asperity spacing *ℓ*, in meters. | No, default: `5e-5` |
| **spectralSlope** | `number` | Exponent *p* of the roughness power spectrum, which varies as *q*^*p* with *q* the spatial frequency. The fractal dimension is *D* = *p*/2 + 2. | No, default: `-1.4` |
| **profile** | `integer` | Accessor of measured surface heights along a track, in meters. | No |
| **sampleSpacing** | `number` | Distance along the surface between consecutive `profile` samples, in meters. Required when `profile` is present. | No |
| **normalTexture** | `object` | Tangent-space normal map giving the surface's mesoscale structure. Defaults to the contacted primitive's material `normalTexture`. | No |
| **material** | `integer` | The index of the acoustic material giving this surface's bulk elastic properties. | No |

σ and *ℓ* together fix the surface's characteristic slope, which scales as σ/*ℓ* and is what a contact actually feels. *p* sets the balance of fine against coarse texture: more negative is smoother-sounding. Representative values for common finishes appear in [Authoring Notes](#authoring-notes).

`material` supplies the elastic constants that set contact stiffness. When absent, the material of the node's modal model applies. A node with neither leaves the contact stiffness undetermined, and implementations MAY use any default.

### Mesoscale Structure

`normalTexture` is a [glTF 2.0 `normalTextureInfo`](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-material-normaltextureinfo), carrying `index`, `texCoord`, and `scale`, and is interpreted exactly as the core material property of the same name: a tangent-space normal map whose X and Y components are scaled by `scale`.

It is sampled along the contact path using the referenced texture coordinate set of the contacted mesh primitive, so it applies only to a node having a mesh with those coordinates. Texel size in meters follows from that primitive's texture coordinate parameterization, which converts the sampled normal into a slope per meter travelled. Because the map is bound to texture coordinates rather than to an absolute size, its features scale with the node, unlike the microscale parameters above.

When `normalTexture` is absent, the [material `normalTexture`](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#_material_normaltexture) of the contacted primitive applies, with its `index`, `texCoord`, and `scale` taken together, so the structure that produces the sound is the structure that is visible. A surface SHOULD set `normalTexture` explicitly only to override that correspondence: to suppress a map whose relief is decorative rather than felt, such as printed graphics, or to supply structure the renderer does not show.

A mesh's materials and texture coordinate sets are both declared per primitive, so both resolve against the primitive containing the contact. Implementations that cannot locate a contact on the mesh MAY use any primitive of that mesh, which is exact for the single-material case.

The two scales add: a contact rides over the mesoscale relief with the microscale finish superimposed. A surface with no `normalTexture` and no material one is smooth at this level and reduces to its statistical finish, correct for a plain machined or cast surface.

### Surface Profiles

`profile` carries a measured height track, sampled uniformly at `sampleSpacing`, relative to the surface's mean plane. Values SHOULD have zero mean. Implementations traverse it cyclically, so authors SHOULD supply a track whose statistics are stationary and whose endpoints are continuous.

A stored profile is the reproducible form: two implementations traversing the same heights produce the same force. A surface without one is specified *statistically* by σ, *ℓ*, and *p*, so implementations agree on its character but not on its detail. Authors requiring identical output across renderers SHOULD store a profile.

With *S* profile samples:

| Property | Accessor Type | Component Type | Count |
|-|-|-|-|
| `profile` | `"SCALAR"` | `5126` (FLOAT) | *S* ≥ 2 |

## Attaching to Nodes

A node participates with a `KHR_audio_rigid_bodies` extension object:

```json
"nodes": [
    {
        "mesh": 0,
        "extensions": {
            "KHR_audio_rigid_bodies": { "modalModel": 0, "acousticSurface": 0, "gain": 1.0 }
        }
    },
    {
        "mesh": 1,
        "extensions": {
            "KHR_audio_rigid_bodies": { "acousticSurface": 1 }
        }
    }
]
```

| | Type | Description | Required |
|-|-|-|-|
| **modalModel** | `integer` | The index of the modal model instanced by this node. | No |
| **acousticSurface** | `integer` | The index of the acoustic surface describing this node's finish. | No |
| **gain** | `number` | Linear amplitude scale applied to this instance's output. | No, default: `1.0` |

Both references are optional, and a node opts into whichever roles apply to it. The first node above both sounds and is scraped. The second, a floor, only supplies its finish to contacts against it. A node with neither is inert.

Multiple nodes MAY reference the same model or surface. Each model instance MUST have independent oscillator state. A node using `EXT_mesh_gpu_instancing` instances the model once per render instance ([Interaction with Other Extensions](#interaction-with-other-extensions)).

## Audio Rendering

A conformant audio renderer synthesizes each model instance's response to contact events as follows. This defines a source *signal* only. How that signal radiates into the scene (directivity, distance, occlusion, reverberation) is out of scope: an implementation MAY layer any radiation or propagation model over it, and absent one SHOULD treat it as a monophonic source at the node's origin.

### Excitation

An excitation is an impulse **j** (in N·s) applied at a surface position **p** at time *t*₀. Both MUST be expressed in the node's local space: **p** by the inverse of the node's global transform, and **j** by the inverse of the global transform's rotation only, preserving the impulse's physical magnitude.

The excitation amplitude of mode *n* is the projection of the impulse onto the mode shape at the contact position:

*a*ₙ = **φ**ₙ(**p**) · **j**

A contact applies force over a finite duration τ, the Hertz contact time, longer for softer and heavier contacts. Each mode is excited by that force's spectrum at its frequency, so implementations SHOULD scale each *a*ₙ by *F̂*(*f*ₙ), the force-pulse spectrum normalized to *F̂*(0) = 1. A half-sine pulse of duration τ is the recommended default: *F̂* is near unity well below 1/(2τ), reaches −3 dB near 1/(2τ), and about −10 dB at 1/τ, so a mode well above 1/τ is barely excited. A resonator bank MAY realize this by driving each mode with the sampled pulse in place of an ideal impulse, which produces the same per-mode scaling without a separate filtering step. That realization generalizes directly to a contact that persists, which [Sustained Contact](#sustained-contact) defines. The impulsive excitation here is the limit of the sustained one as the contact duration goes to zero.

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

### Sustained Contact

An impact excites the modes transiently. A contact that persists excites the same modes continuously as the irregularities of the two [acoustic surfaces](#acoustic-surfaces) ride over one another, imposing a relative displacement at the contact. Modes, mode shapes, and mass properties are unchanged; only the driving force differs.

This mechanism is **roughness excitation**, and two fields converge on it. Rolling-noise prediction for railways and tyres treats wheel and rail roughness as the excitation (Remington 1987, Thompson et al. 2003). Friction acoustics calls it *roughness noise*, produced by asperity impacts in light contact, and separates it from the instability-driven noise of strongly loaded contact (Akay 2002).

It is therefore distinct from impact excitation, where contact makes and breaks as a single event, and from friction-induced vibration, which is self-excited. This section defines roughness excitation only; the other two are [Excitation](#excitation) and an exclusion respectively. Sliding, scraping, and rolling are regions of one continuous parameter space, not separate mechanisms.

Implementations SHOULD render sustained contact. A body supplying no acoustic surface contributes the default values, so a contact is always fully specified.

#### Contact State

At each instant, a sustained contact on a body is described by six quantities:

| | Description |
|-|-|
| **p** | Contact position on the body's surface. |
| **n̂** | Unit contact normal, directed into the body. |
| *N* | Normal force at the contact, in newtons. Non-negative. |
| **u** | Sweep velocity: the velocity of the contact position over this body's own surface. **Each body has its own**, and they are generally unequal. |
| **u**<sub>slip</sub> | Slip velocity: the velocity of the other body's material point at the contact relative to this body's. The frictional force on this body acts along it. |
| *μ* | Friction coefficient of the pair, combined as [KHR_physics_rigid_bodies](#interaction-with-other-extensions) defines. |

Both velocities are vectors, because the [contact force](#contact-force) has a direction in the contact plane and not only a magnitude there. Their magnitudes *v*<sub>sweep</sub> = |**u**| and *v*<sub>slip</sub> = |**u**<sub>slip</sub>| are the speeds the requirements below are written in terms of.

How a host obtains these is out of scope, as contact reporting is for [Excitation](#excitation).

The two bodies' kinematics relate only in a frame they share, and which frame that is does not matter. Writing **u**₁ and **u**₂ for the two sweep velocities in any one common frame, body 1's slip velocity is **u**₁ − **u**₂ and body 2's is its negation, so the slip speed |**u**₁ − **u**₂| is shared between them while its direction is not. Together these distinguish every regime:

- **Pure rolling**: both sweeps nonzero and equal, so slip vanishes. Nothing slides, yet the contact sweeps both surfaces.
- **A box sliding on a fixed floor**: zero sweep on the box, since the same material region stays in contact, and sweep on the floor equal to the slip.
- **Partial slip** lies between.

Implementations SHOULD NOT select between separate rolling and sliding models, which introduces an audible transition where the physics has none. Because the sweeps differ, each body's surface is traversed at its own rate and the contributions summed ([Composite Surface](#composite-surface)); a body whose own sweep is zero still sounds, because the other surface streams past its stationary patch.

A contact is rendered independently for each body that instances a modal model, each driven by its own contact state with opposed normals. Each body's state MUST be expressed in that body's node-local space, by the same rule the impulsive [Excitation](#excitation) uses: **p** by the inverse of the node's global transform, and **n̂**, **u**, and **u**<sub>slip</sub> by the inverse of its rotation only, preserving their physical magnitudes. A velocity keeps its magnitude because an absolute surface finish is traversed at a rate the world sets: `sampleSpacing` does not scale with the node ([Node Transforms and Scale](#node-transforms-and-scale)), so scaling the velocity instead would read the finish at the wrong rate.

#### Composite Surface

A contact rides over both surfaces at once. Writing subscripts 1 and 2 for the two bodies, implementations SHOULD combine them as

σ\* = √(σ₁² + σ₂²),  *ℓ*\* = (*ℓ*₁ + *ℓ*₂)/2,  *p*\* = (*p*₁ + *p*₂)/2

The first is the standard composite roughness of two surfaces in contact. When exactly one body supplies a `profile`, implementations SHOULD traverse that profile. When both do, they SHOULD sum them. When neither does, they SHOULD synthesize a track with root-mean-square height σ\*, correlation length *ℓ*\*, and spectral slope *p*\*.

Each contribution advances at **that surface's own sweep speed**, and the two are summed to give the gap between the bodies, so two profiles are read at two independent positions rather than as one track at one rate. A single synthesized track standing in for both follows whichever sweep is faster: an approximation, since one track cannot carry two rates, but it keeps the dominant one and is exact whenever one surface is at rest relative to the contact, which covers both a fixed floor and pure rolling.

Mesoscale structure combines the same way. Each body's `normalTexture` is sampled along its own contact path at its own sweep speed and the relief contributions add; a body without one contributes nothing at that scale.

#### Contact Force

Two quantities need distinguishing. The *contact force* is the physical force the bodies exert on one another, with a part along **n̂** that carries the static load and is never negative, and a part in the contact plane. The *excitation* **F**(*t*) is what drives the modes: the fluctuation of that force about its equilibrium value, a vector taking either sign in every component and averaging to zero.

The tangential part arises by two mechanisms, which act along different directions. The *geometric* part is the contact load projected onto the locally tilted surface, so it acts along the direction the contact travels over that surface and is present whenever a surface sweeps, rolling included. The *frictional* part is Coulomb traction, so it acts along the direction of slip and vanishes when nothing slides.

This specification constrains their behavior rather than their formula, so that assets sound consistent across implementations without dictating a contact model. [Appendix B](#appendix-b-reference-contact-force-model) gives one model consistent with the requirements below.

- The contact force MUST be non-negative: a contact that separates applies no force. This nonlinearity is what produces micro-collisions and chatter.
- The two bodies MUST receive equal and opposite excitations. **n̂** and **u**<sub>slip</sub> are defined per body and reverse between them, so the normal and frictional parts reverse with the [contact state](#contact-state). The sweep directions are shared between the bodies, so the geometric part reverses by surface instead: a body's own surface acts along **û**ᵢ and the other body's acts against it.
- A contact at rest MUST produce no output. With the slip speed and both sweep speeds zero and *N* constant, the excitation is zero, so a settled body is silent however heavily it is loaded.
- Each surface's track SHOULD be indexed by the **distance the contact has travelled along that surface**, not by time. Indexing by each surface's *own* distance is what keeps a body sliding on a fixed floor audible, since its own sweep is zero while the floor's equals the slip. A track is one dimensional and a contact path is not, so a contact retracing its path reads fresh surface rather than the surface it came from ([Scope and Exclusions](#scope-and-exclusions)).
- Loudness SHOULD increase with normal force and with the rate surface passes through the contact. Every component scales with the load: the normal one through the contact stiffness, the geometric tangential one because it is the load projected onto a tilted surface, and the frictional one because Coulomb traction is bounded by *μN*. None of them carries an explicit speed factor. Speed enters by scaling the spectrum of the traversal, which is what makes a rolling body grow louder with speed although it has no slip.
- Roughness with wavelengths shorter than the contact patch SHOULD be attenuated. This is the **contact filter**, standard in rolling-noise prediction since Remington 1987, and it acts on spatial wavelength rather than on time: a patch of radius *a* cannot resolve wavelengths below roughly 2*a* at any speed. Applied to the track rather than to the output, its audible cutoff tracks speed for free, landing near *v*<sub>sweep</sub>/(2*a*). Treat that as a scale, not a corner frequency, since measured filters roll off gradually from a factor of a few below it. Because *a* grows with load, the filter also softens under heavier contact.
- The roughness-driven part of the force SHOULD be limited by the applied load, which a contact cannot exceed. Without the limit the result is audibly rougher than the surface it models.

#### Exciting the Modes

The contact drives the modes as a force rather than an impulse. The excitation amplitude of mode *n* is the projection of the excitation force **F** onto the mode shape at the current contact position:

*a*ₙ(*t*) = **φ**ₙ(**p**(*t*)) · **F**(*t*)

A modal renderer already driving each mode with a sampled force pulse ([Excitation](#excitation)) renders this by substituting **F**(*t*) for the impulse, with no other change.

**F** MUST be applied as a vector, its normal part along **n̂**, its geometric tangential part along each surface's own sweep direction, and its frictional part along the slip direction. Collapsing it to a scalar along **n̂** would make a tangential scrape and a normal press excite the modes identically, the direction blindness that [mode shapes](#sample-points) are vectors to avoid, and sliding surfaces are observed to develop contact forces in both directions, each driving its own response (Akay 2002). Only the frictional part vanishes with slip, so pure rolling still develops a tangential force and still needs a direction to apply it along.

The contact position is a function of time, and evaluating **φ**ₙ at the current position rather than holding it fixed for the duration of the contact varies the timbre as a body is dragged across its surface. Implementations SHOULD evaluate it continuously, using the barycentric interpolation of [Sample Points](#sample-points) when the model supplies `indices` and blending between the nearest sample points otherwise. Nearest-point evaluation alone leaves the field piecewise constant over the sample points' Voronoi cells, which is acceptable for an impact but steps audibly each time a sustained contact crosses a cell boundary.

Because only the shape gains vary with position and the frequencies and decay rates are shared, this interpolation is exact in the sense that it never detunes a mode.

#### Vibrational Coupling

The separation between two surfaces in contact is modulated by the body's own vibration, so the contact force and the modal state form a feedback loop. Writing *u*(*t*) for the modal surface displacement along the contact normal,

*u*(*t*) = Σₙ (**φ**ₙ(**p**) · **n̂**) *q*ₙ(*t*)

with *q*ₙ the displacement of mode *n*, the separation driving the contact is the rigid approach less *u*(*t*).

Implementations MAY couple the contact force to *u*(*t*). Coupling produces micro-collisions, chattering, and the contact-dependent damping of a body pressed against another, none of which an open-loop force reproduces, and costs only a per-mode read of resonator state that any bank already has. The effect is strongest for rolling: analysing recordings, van den Doel et al. 2001 found the rolling force coupled to the modes strongly enough that an independent force driving a linearly vibrating object no longer described their measurements.

Coupling makes the force generator nonlinear, but the mode bank stays linear, so the requirement that excitations superpose linearly ([Synthesis](#synthesis)) is unaffected. When both bodies instance modal models a full treatment shares one separation between both banks; implementations MAY instead couple each body only to its own vibration.

## Node Transforms and Scale

A modal model describes its object at a fixed physical size: the node's global transform in the scene's initial state (before any animation). Sample point positions transform with the node like mesh vertices.

Vibration frequencies are not invariant under scaling. If the node's global scale is later changed *uniformly* by a factor γ relative to the initial state, implementations SHOULD adjust the model: with each mode's undamped angular frequency ωₙ = √((2π*f*ₙ)² + *d*ₙ²), scale ωₙ → ωₙ/γ and **φ**ₙ → γ⁻³ᐟ² **φ**ₙ, recompute *d*ₙ from the material's damping function evaluated at the scaled ωₙ (otherwise leave *d*ₙ unchanged), and derive *f*ₙ = √(ωₙ² − *d*ₙ²)/2π. For a material carrying `alpha` and `beta` that function is *d* = (α + βω²)/2. Implementations that do not support rescaling SHOULD render the model unmodified.

Behavior is undefined whenever the node's global transform contains non-uniform scale: non-uniform scaling changes the object's mode shapes and frequencies in ways that cannot be recovered from precomputed data (and leaves the transform's rotation ill-defined). Authors requiring a non-uniformly scaled object bake the scale into the geometry and analyze the result.

A surface's microscale quantities are exempt. `roughness`, `correlationLength`, `profile`, and `sampleSpacing` are absolute physical lengths and MUST NOT be scaled by the node transform, because a finish does not change when an object is resized: a scaled-up polished sphere is still polished. Contact position and sweep velocity are geometric and do transform.

`normalTexture` is the one surface quantity that does scale. It carries no absolute size, only a size in texture coordinates, so its features measure whatever the primitive's parameterization and the node's global scale make them: a scaled-up tiled floor has larger tiles, audibly as well as visibly. Implementations MUST derive its texel size in meters from the node's global scale in force at the contact, not from the scale the mesh was authored at, so that two nodes instancing one mesh at different sizes do not share one relief.

## Interaction with Other Extensions

**KHR_physics_rigid_bodies**: This extension is that one's acoustic counterpart, over the same bodies. Although excitations may come from any host source, the typical source is a rigid-body simulation, where a collision yields the two quantities an impulsive excitation requires: a contact impulse and a world-space contact position, as in that extension's `rigid_body/applyPointImpulse` interactivity node. A persisting contact supplies the [contact state](#contact-state) instead, every quantity of which a rigid-body solver computes in the course of resolving the contact. Contacts on a collider SHOULD excite the modal model, and use the acoustic surface, of the collider node or its nearest ancestor that has one. A rigid body sounds as one elastic object, so at most one node of a body's hierarchy (the body node and its collider nodes) MUST carry a modal model. Each collider MAY carry its own acoustic surface. Any local surface geometry a contact model reads, such as curvature, SHOULD come from the mesh of the collider node it landed on, or from that of its nearest ancestor with a mesh. A collider's geometry and its acoustic surface resolve independently, so a collider carrying a mesh supplies that geometry whether or not it also carries a surface. The mechanism by which a simulation reports contact events is beyond the scope of this specification.

The dissipative part of the [contact force](#contact-force) SHOULD be derived from the physics material's `restitution` when one is present, bearing in mind that restitution varies with approach speed while the contact model's dissipation constant does not ([Appendix B](#appendix-b-reference-contact-force-model)). This extension deliberately does not duplicate restitution or friction, which have a standardized home there and no acoustic reading distinct from their mechanical one. The [contact state](#contact-state)'s friction coefficient is likewise that extension's, combined by its own combine modes rather than restated here.

**EXT_mesh_gpu_instancing**: A node using GPU instancing instantiates its modal model once per render instance, each with independent oscillator state and the node's `gain`. An excitation targets a single render instance. Attribution is host logic, like contact reporting. Excitation mapping ([Excitation](#excitation)) uses the composed instance transform (the transform applied to the instance's vertices for rendering, as defined by `EXT_mesh_gpu_instancing`) in place of the node's global transform, and each instance's output is a monophonic source at that instance's origin. The model describes the object under identity instance transform. An instance's uniform scale relative to that reference (its `SCALE` attribute, composed with any node scale change) is a uniform scale change under [Node Transforms and Scale](#node-transforms-and-scale), and behavior is undefined when the composed transform contains non-uniform scale.

**KHR_audio_emitter / KHR_audio_graph** (proposals): This extension generates source signals and does not define emitters, listeners, or spatialization. When a node with a modal model also has an audio emitter, the synthesized signal SHOULD be routed to that emitter as a source.

An acoustic material is distinct from a physics material (friction, restitution) and a render material. A node may carry all three.

An acoustic surface's `roughness` is distinct from a render material's `roughness`, despite the shared word. That parameter describes optical microfacet statistics, is dimensionless, and is tuned by eye. This one is a physical length, is measured with a profilometer, and describes structure orders of magnitude coarser. Neither can be derived from the other.

An acoustic surface's `normalTexture` is the opposite case: it is the same kind of data as the core material property, uses the same [`normalTextureInfo`](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-material-normaltextureinfo) definition, and defaults to the very same reference, so a contact sounds like the surface it is visibly crossing. The surface property overrides that default rather than establishing it.

## Scope and Exclusions

The following are deliberately out of scope. Each composes *with* the modal core rather than replacing it, and may be layered by future extensions:

- **Acoustic radiation transfer** (listener-position-dependent, per-mode amplitude fields, e.g. FFAT maps or multipole expansions). Without it, a model radiates omnidirectionally, including [acceleration noise](#acceleration-noise) directivity.
- **Sound propagation and spatialization**: distance attenuation, occlusion, reverberation, and listener modeling belong to audio-emitter and platform layers.
- **Contact-event plumbing**: how a physics engine reports impulses and contact state to the audio system is application logic.
- **Friction-induced vibration**: squeal, bowing, and brake noise. Of the three mechanisms that excite a body in contact, this extension defines two, impact excitation and roughness excitation, and leaves out the third. Friction-induced vibration is self-excited, arising when velocity-weakening friction drives a limit cycle under strongly loaded contact, rather than imposed by surface irregularity, and it needs its own contact state to express. It composes as an additional force term on the same contact.
- **Anisotropic and spatially varying microscale finish**: `roughness`, `correlationLength`, and `spectralSlope` are isotropic and uniform over the body, so a finish that is directional below the visible scale sounds the same scraped along the grain as across it. Variation and directionality at the mesoscale are carried by `normalTexture`, which a contact samples along its own path.
- **Position-indexed traversal of the microscale finish**: `profile` is a one-dimensional track and σ, *ℓ*, *p* describe a one-dimensional realization, so the finish is traversed by distance along the contact path and a contact retracing its path reads fresh finish rather than what it came from. A rocking body hisses where it should rattle. Making the finish a field over the surface is feasible with stored data over an extent matched to the contact's excursion, as Agarwal et al. 2021 do with a measured two-dimensional depth map for a hand-held scraper. Covering unbounded sliding instead needs, for *T* seconds of non-repeating output at the output Nyquist frequency, on the order of *T* *f*<sub>Nyquist</sub> samples in one dimension and the square of that in two, independent of speed, since a slower contact needs proportionally finer detail over a proportionally smaller extent. Procedural evaluation carries no such cost and synthesizes only the octaves the current speed and load resolve, which is the route a future extension would take. Every one-dimensional projection that recovers retracing instead makes the excitation depend on sliding direction, which the surface does not. **The mesoscale layer is not subject to this**, since `normalTexture` is already a field over the surface and the [contact state](#contact-state) already carries the contact position **p** to sample it at.
- **Nonlinear vibration**: mode coupling in thin shells (cymbals, sheet metal) and fracture. Contact-dependent damping is reachable through [vibrational coupling](#vibrational-coupling) but is not otherwise modeled.
- **Recorded-sample hybrids**: mixing recorded impact audio with the synthesized modes for detail beyond linear modal synthesis.
- **Modal analysis itself**: meshing, FEM, and eigensolves happen at authoring time ([Appendix A](#appendix-a-deriving-modal-data)).

## Authoring Notes

*This section is non-normative.*

Linear modal models are valid for near-rigid solid bodies under small deformations. Thin shells and strongly nonlinear objects are poorly reproduced. Deriving modes with FEM requires a watertight, tet-meshable solid at authoring time, but this extension imposes no geometry requirements at runtime. Sample points are self-contained, so render meshes may be remeshed, decimated, or LOD-swapped without invalidating the audio data.

Modes are typically band-limited to [20 Hz, 20 kHz] at authoring time. A few tens of modes are perceptually sufficient for most objects, and a few hundred for large or broadband ones. Sample point density is an authoring choice: a handful of points suffices for coarse position-dependent timbre, while per-vertex density captures fine spatial variation at proportional cost. Sample spacing matters more for sustained contact than for impacts, since a contact travelling across a sparse point set hears the interpolation rather than the object.

The two scales differ in more than size. Rolling-noise work finds that for ordinary surface roughness a linear treatment predicts the sound well, while discrete features large enough to unload the contact make the nonlinearity significant (Thompson et al. 2003). The microscale is therefore forgiving, and the mesoscale is where the separation clamp of [Contact Force](#contact-force) earns its place: a contact crossing a grout line or a tread edge can momentarily lift, and a model that cannot express that will sound smooth over exactly the features meant to be heard.

Deciding which scale a feature belongs to is mostly a question of whether it is visible. Tiling, grout, corrugation, knurling, tread, and grain are mesoscale: they are millimetres or larger, they are already in the asset as a normal map, and they dominate the sound of a body rolling over them. The finish between those features is microscale. A surface with visible structure and no `normalTexture` will sound plausible but featureless, since the collision geometry does not carry that structure and the statistical parameters describe something far finer.

Representative surface finishes, for authoring the microscale by name rather than by measurement:

| Finish | σ (m) | *ℓ* (m) | *p* | *D* |
|-|-|-|-|-|
| Polished | 1e-7 | 1e-5 | -1.8 | 1.1 |
| Machined | 2e-6 | 5e-5 | -1.4 | 1.3 |
| Sandblasted | 1e-5 | 1e-4 | -1.2 | 1.4 |
| Cast or unfinished | 1e-4 | 1e-3 | -1.0 | 1.5 |

These are illustrative starting points rather than measurements. The one anchored figure is the machined row: fractal dimensions between 1.17 and 1.39 are reported for various machined surfaces at a length scale of 10⁻⁶ m, in the surface-metrology literature surveyed by van den Doel et al. 2001. The fractal model holds over a wide but not unlimited band. Fitting the power spectrum of a scrape on smooth plastic, the same work found it valid to about 1 kHz.

Surfaces may be characterized two ways. A profilometer or scanning confocal microscope yields a height track directly, which is what `profile` carries, at a horizontal resolution of a few microns (Agarwal et al. 2021 measure at 5.6 µm). Track length follows from how fast it is consumed: a contact sweeping at speed *v* reads *v*/`sampleSpacing` samples per second, so at 5 µm spacing and 10 cm/s a one-second loop needs roughly 20,000 samples. Tens of thousands of samples, a few hundred kilobytes, keeps the repetition below notice at ordinary speeds. A contact microphone recording of a scrape yields *p* from a linear fit to the power spectrum, leaving σ and *ℓ* to be matched by ear against the recording.

## JSON Schema

- [glTF.KHR_audio_rigid_bodies.schema.json](schema/glTF.KHR_audio_rigid_bodies.schema.json)
- [glTF.KHR_audio_rigid_bodies.model.schema.json](schema/glTF.KHR_audio_rigid_bodies.model.schema.json)
- [glTF.KHR_audio_rigid_bodies.material.schema.json](schema/glTF.KHR_audio_rigid_bodies.material.schema.json)
- [glTF.KHR_audio_rigid_bodies.surface.schema.json](schema/glTF.KHR_audio_rigid_bodies.surface.schema.json)
- [node.KHR_audio_rigid_bodies.schema.json](schema/node.KHR_audio_rigid_bodies.schema.json)

## Object Model

The following JSON pointer is defined for use with the glTF Object Model (e.g. by `KHR_animation_pointer` and `KHR_interactivity`):

| Pointer | Type |
|-|-|
| `/nodes/{}/extensions/KHR_audio_rigid_bodies/gain` | `float` |

Modal model, acoustic material, and acoustic surface data are static and not addressable.

## Known Implementations

- [MeshEditor](https://github.com/khiner/MeshEditor) — authoring (FEM modal analysis from meshes) and rendering (coupled-form resonator bank), with extension import/export in progress.

## References

- J. A. Greenwood, J. B. P. Williamson. *Contact of Nominally Flat Surfaces.* Proc. R. Soc. Lond. A 295, 1966.
- G. Maidanik. *Energy Dissipation Associated with Gas-Pumping in Structural Joints.* J. Acoust. Soc. Am. 40, 1966.
- K. H. Hunt, F. R. E. Crossley. *Coefficient of Restitution Interpreted as Damping in Vibroimpact.* Journal of Applied Mechanics 42(2), 1975.
- K. L. Johnson. *Contact Mechanics.* Cambridge University Press, 1985.
- P. J. Remington. *Wheel/Rail Rolling Noise, I: Theoretical Analysis.* J. Acoust. Soc. Am. 81(6), 1987.
- K. van den Doel, D. K. Pai. *The Sounds of Physical Shapes.* Presence 7(4), 1998.
- A. Akay. *Acoustics of Friction.* J. Acoust. Soc. Am. 111(4), 2002.
- D. Thompson, T. Wu, T. Armstrong. *Wheel/Rail Rolling Noise: The Effects of Non-Linearities in the Contact Zone.* 10th International Congress on Sound and Vibration, 2003.
- K. van den Doel, P. G. Kry, D. K. Pai. *FoleyAutomatic: Physically-based Sound Effects for Interactive Simulation and Animation.* SIGGRAPH 2001.
- J. F. O'Brien, C. Shen, C. M. Gatchalian. *Synthesizing Sounds from Rigid-Body Simulations.* SCA 2002.
- D. L. James, J. Barbič, D. K. Pai. *Precomputed Acoustic Transfer: Output-sensitive, Accurate Sound Generation for Geometrically Complex Vibration Sources.* SIGGRAPH 2006.
- C. Zheng, D. L. James. *Rigid-Body Fracture Sound with Precomputed Soundbanks.* SIGGRAPH 2010.
- B. N. J. Persson. *Relation between Interfacial Separation and Load: A General Theory of Contact Mechanics.* Phys. Rev. Lett. 99, 2007.
- Z. Ren, H. Yeh, M. C. Lin. *Synthesizing Contact Sounds Between Textured Models.* IEEE VR 2010.
- H. Ben Abdelounis, A. Le Bot, J. Perret-Liaudet, H. Zahouani. *An Experimental Study on Roughness Noise of Dry Rough Flat Surfaces.* Wear 268, 2010.
- A. Le Bot, E. Bou Chakra. *Measurement of Friction Noise versus Contact Area of Rough Surfaces Weakly Loaded.* Tribology Letters 37, 2010.
- A. Le Bot, E. Bou-Chakra, G. Michon. *Dissipation of Vibration in Rough Contact.* Tribology Letters 41, 2011.
- V. H. Hung, J. Perret-Liaudet, J. Scheibert, A. Le Bot. *Direct Numerical Simulation of the Dynamics of Sliding Rough Surfaces.* Computational Mechanics 52, 2013.
- C. Zheng, D. L. James. *Toward High-Quality Modal Contact Sound.* SIGGRAPH 2011.
- J. N. Chadwick, C. Zheng, D. L. James. *Precomputed Acceleration Noise for Improved Rigid-Body Sound.* SIGGRAPH 2012.
- T. R. Langlois, S. S. An, K. K. Jin, D. L. James. *Eigenmode Compression for Modal Sound Models.* SIGGRAPH 2014.
- A. Sterling, M. C. Lin. *Interactive Modal Sound Synthesis Using Generalized Proportional Damping.* I3D 2016.
- A. Le Bot. *Noise of Sliding Rough Contact.* J. Phys.: Conf. Ser. 797, 2017.
- A. Papangelo, N. Hoffmann, M. Ciavarella. *Load-Separation Curves for the Contact of Self-Affine Rough Surfaces.* Scientific Reports 7(1), 2017.
- J.-H. Wang, D. L. James. *KleinPAT: Optimal Mode Conflation for Time-Domain Precomputation of Acoustic Transfer.* SIGGRAPH 2019.
- V. Agarwal, M. Cusimano, J. Traer, J. H. McDermott. *Object-Based Synthesis of Scraping and Rolling Sounds Based on Non-Linear Physical Constraints.* DAFx 2021.
- S. Clarke et al. *RealImpact: A Dataset of Impact Sound Fields for Real Objects.* CVPR 2023.
- X. Jin, C. Xu, R. Gao, J. Wu, G. Wang, S. Li. *DiffSound: Differentiable Modal Sound Rendering and Inverse Rendering for Diverse Inference Tasks.* SIGGRAPH 2024.
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

Rayleigh damping is a two-parameter special case of a broader family, and cannot represent every material: its stiffness term fixes the exponent relating damping to frequency, while measured attenuation exponents vary by material (Sterling & Lin 2016). Whether richer damping models sound better is less settled, and that same work found its power-law model no more realistic than Rayleigh in a listening test.

Per-mode `decayRates` make the question moot for interchange, since they represent any damping distribution exactly. This is also what current practice fits: differentiable modal renderers learn a damping factor per mode rather than deriving one from a material (Jin et al. 2024). The one place the Rayleigh form is applied at render time is re-deriving damping after a uniform rescale. An author whose decay rates were fitted rather than derived SHOULD omit `alpha` and `beta`, which leaves the measured values untouched.

Models may also be fitted from measured impact recordings (e.g. spectral peak picking for frequencies, log-envelope regression for decay rates, per-strike-point amplitudes for shapes), in which case decay rates are unconstrained by the Rayleigh model and radiation effects may be baked into the shapes.

## Appendix B: Reference Contact Force Model

*This appendix is non-normative.*

One model satisfying every requirement in [Contact Force](#contact-force), assembled from the literature. Implementations are free to use any other.

The contact's elastic constants come from the two bodies' acoustic materials and the local surface curvature, and are the same constants that set the Hertz contact time of an impact. Hertz theory for the normal contact of elastic solids (Johnson 1985) gives:

1/*E*\* = (1 − ν₁²)/*E*₁ + (1 − ν₂²)/*E*₂,  *R*\* = 1/(κ₁ + κ₂),  *k* = (4/3) *E*\* √*R*\*

with κ the mean surface curvature of each body at the contact point, taken from the geometry of the collider the contact landed on. This form is exact where the contact is axisymmetric, as it is for a sphere against a sphere or a plane. Where the two principal curvatures differ the contact patch is elliptical, and κ₁ + κ₂ is the sum term of that solution rather than its whole. The contact patch radius follows as *a* = (3*N* *R*\*/(4*E*\*))^(1/3), which sets the patch-filter cutoff, and the equilibrium penetration under the current load is δ₀ = (*N*/*k*)^(2/3).

Then, each sample:

- Hold one read position per surface, each advancing by the distance that surface has travelled through the contact, so traversal is indexed by distance rather than time. In samples it advances by that surface's sweep speed divided by the track's sample spacing: `sampleSpacing` for a stored profile, a synthesis parameter otherwise.
- Form each surface's relief *h*ᵢ = *h*<sub>meso,ᵢ</sub> + σᵢ *h*<sub>micro,ᵢ</sub>, then the gap relief *h* = *h*₁ + *h*₂. The microscale term is the track sample at that read position. The mesoscale term integrates the slope implied by that surface's `normalTexture` along its path, and is zero without one; since a sampled normal map is not exactly a gradient field, use a leaky integrator to keep the accumulated drift out. The tangential term below wants each surface's slope directly and needs no integration. A surface with zero sweep contributes a constant, removed by the equilibrium subtraction.
- Apply the contact filter by smoothing each track over a window of about 2*a* **in distance along the surface**, not in time; speed dependence then follows from the traversal, landing near *v*<sub>sweep,ᵢ</sub>/(2*a*) in the output. The window widens with load as *a* grows, which is why rolling-noise work moves this filter into the time domain once deflections stop being small (Thompson et al. 2003). Here the two scales separate: millimetre relief passes intact while a micron finish sits at or below the patch and is strongly attenuated, so rolling hears the tiles and not the polish with no rolling-specific branch.
- Form the rigid approach δ<sub>rigid</sub> = δ₀ + *h*, and the true separation δ = max(δ<sub>rigid</sub> − *u*, 0) with *u* the modal displacement of [Vibrational Coupling](#vibrational-coupling), or zero without coupling. The clamp at zero is the separation nonlinearity.
- Normal force by the Hunt and Crossley model, *f*<sub>n</sub> = *k* δ^(3/2) (1 + *c*<sub>d</sub> δ̇), with *c*<sub>d</sub> = (3/2)α. Here α is a material constant relating restitution to approach speed by *e* ≈ 1 − α*v*ᵢ at low speed, reported between 0.08 and 0.32 s·m⁻¹ for steel, bronze, and ivory. Being a material property rather than a property of one impact, it needs no reference speed; recovering it from a physics material's `restitution` does need the speed that value was quoted at.
- Tangential force in two parts, neither of which carries a free parameter. The geometric part is the normal force projected onto the locally tilted surface, *f*<sub>t,geo,ᵢ</sub> = *f*<sub>n</sub> ∂*h*ᵢ per surface, with ∂*h*ᵢ the slope of that surface's own relief along its own path. The frictional part is Coulomb traction riding on the force fluctuation, *f*<sub>t,fric</sub> = *μ*(*f*<sub>n</sub> − *N*). Both are fluctuations about equilibrium, the frictional part by construction and the geometric part by removing its local mean the same way the relief does, so a surface at rest sits on a constant slope and excites nothing. Agarwal et al. 2021 write one term instead, *f*<sub>h</sub> = β₁|**v** · ∇*S*|^β₂ with β₁ = 0.05 and β₂ = 1, where **v** is the contact's velocity across the surface rather than the slip, so their term survives pure rolling exactly as the geometric part here does. That term is the same projection's velocity-dependent half: because *f*<sub>n</sub> carries the Hunt and Crossley damping factor, *f*<sub>n</sub> ∂*h* expands to *k*δ^(3/2) ∂*h* + *k*δ^(3/2) *c*<sub>d</sub> δ̇ ∂*h*, and the second part is proportional to the rate the height under the contact changes. Matching the two identifies β₁ with *f*<sub>n</sub> *c*<sub>d</sub> ∂*h*, which carries exactly the N·s·m⁻¹ their constant needs and shows what a fitted value absorbs: the load, the material's dissipation constant, and the surface's slope. Projecting the whole contact force recovers both halves with nothing left to fit.
- Excite with **F** = **n̂**(*f*<sub>n</sub> − *N*) + Σᵢ *s*ᵢ **û**ᵢ *f*<sub>t,geo,ᵢ</sub> + **t̂** *f*<sub>t,fric</sub>, with **û**ᵢ the unit sweep direction of surface *i*, **t̂** the unit slip direction, and *s*ᵢ the reversal [Contact Force](#contact-force) requires: +1 for the body's own surface and −1 for the other body's. Subtracting the load is exact where a high-pass would only approximate it and would color the low modes. Agarwal et al. 2021 sum the components as scalars, which suffices for one measured impulse response but discards direction against vector mode shapes.
- Soft-limit the roughness-driven part of *f*<sub>n</sub> with a knee that scales with *N*, which bounds the force to what the load sustains. Agarwal et al. apply the equivalent limit to the trajectory curvature instead, using a tanh nonlinearity whose parameters vary with normal force.

Rolling needs no separate term: *v*<sub>slip</sub> vanishes so the frictional part drops out, while both sweeps do not, so the geometric part and the normal channel keep traversing both surfaces. Agarwal et al. 2021 add a rolling term from the offset between a ball's center of mass and its geometric center, omitted here because it presumes a sphere and the patch-filtered relief already gives rolling its coarser character on arbitrary geometry.

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
