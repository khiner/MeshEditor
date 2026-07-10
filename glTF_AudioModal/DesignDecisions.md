# KHR_audio_modal — Design Decisions and Rationale

Companion reference for the spec in `extensions/2.0/Khronos/KHR_audio_modal/`. Not part of the specification. Records what was decided, why, what was rejected, and what would prompt revisiting.

## Name and positioning: `KHR_audio_modal`

**Decision.** KHR-prefixed name, Draft status, `extensions/2.0/Khronos/` layout — the glTF_Physics playbook (KHR-named draft staged for ratification, e.g. `KHR_physics_rigid_bodies`).

**Why.** Khronos naming convention is `<PREFIX>_<scope>_<feature>`; `audio` is the scope already established by the `KHR_audio_graph` / `KHR_audio_emitter` proposals. Naming into that scope signals the intended relationship: those proposals cover sources/emitters/listeners/spatialization; this extension is a complementary signal-*generating* layer they can consume. No modal or procedural-audio glTF proposal exists anywhere (verified across Khronos, OMI, MPEG, middleware), so this is a green field with no name collision.

**Ratification strategy.** The realistic ladder is vendor prefix → EXT (second independent implementation) → KHR (Khronos vote). The KHR name is permitted pre-ratification when that intent exists (glTF_Physics precedent). Review Draft stage requires JSON schema + test assets + a third-party implementation; test assets are deliberately deferred until the spec settles.

## One extension, not a data/behavior split

**Decision.** A single extension carries both the data (models, materials) and the rendering semantics.

**Why.** glTF_Physics split `KHR_implicit_shapes` (data-only) from `KHR_physics_rigid_bodies` (behavior) because the shape data has multiple prospective consumers. Modal model data has exactly one consumer — a modal audio renderer — so a split would add ceremony without reuse. Revisit if a second consumer materializes (e.g. a haptics extension wanting the same modes).

## Root-level arrays + node index reference

**Decision.** `models[]` and `materials[]` live in a `KHR_audio_modal` object on the glTF root (items inherit `glTFChildOfRootProperty`, so they get `name`); nodes instance a model by index with an optional per-instance `gain`.

**Why.** This is the established registry pattern (`KHR_lights_punctual` lights, physics `physicsMaterials`): reusable resources defined once, instanced by many nodes. Ten identical bowls share one model; each node instance has independent oscillator state (normative), which is the audio analog of instanced rendering. `gain` sits on the node, not the model, because it is an instance-level mixing knob (same model, quieter instance); JASS/Phya global scale factors are the precedent.

## Full 3-vector mode shapes, not scalar gains

**Decision.** The canonical spatial payload is per-mode, per-sample-point *displacement vectors* φ (mass-normalized eigenvectors). Excitation is the projection aₙ = φₙ(p) · j.

**Why.** Scalar per-position gains (MeshEditor's current `|φ|`, max-normalized per position) are direction-blind: a tangential scrape and a normal strike at the same point excite identically, which is physically wrong. Vectors preserve impulse-direction-dependent timbre and additionally make normal-projected surface velocity derivable — the quantity every radiation model consumes (WaveBlender reduces to exactly this at runtime). Scalar gains are derivable from vectors; the reverse is lossy. The literature's full-fidelity systems (O'Brien 2002, all Zheng/James, ModalSound, WaveBlender) all keep vectors.

**Cost accepted.** 3× the data of scalars. Langlois et al. 2014 show mode-shape fields compress ~100:1 transparently if size ever matters; a compression extension can layer on later.

**Implication.** MeshEditor's `mesh2modes` must be upgraded to retain eigenvector 3-vectors (known follow-up).

## Self-contained sample points, not mesh-vertex binding

**Decision.** Mode shapes are sampled at an explicit point set (`positions`, node-local), independent of any mesh. A node with a model need not have a mesh. The point set is, formally, just a point cloud that transforms rigidly with the node.

**Why.**
- There is no canonical "the mesh" to bind to: shapes come from a tet mesh whose vertices don't correspond to render-mesh vertices; render meshes have split vertices (normal/UV seams), multiple primitives with independent vertex numbering, and get decimated or LOD-swapped. Vertex-index binding inherits all of that ambiguity.
- Contacts arrive as *positions*, not vertex indices — physics contacts occur on collider geometry (hulls, implicit shapes), and the interop currency (`rigid_body/applyPointImpulse`) is a world-space position + impulse. A point cloud queried by position is the natural consumer.
- Placement correctness is an authoring responsibility, like normals or skin weights. The semantics (evaluate the sampled field at the query point) stay well-defined regardless of placement; there is no normative surface to validate against anyway. Interior points are even physically meaningful (modes are volumetric fields); surface sampling is the useful convention because contacts are surface events.
- Sampling density is an authoring dial: a handful of points (RealImpact-style strike points) up to per-vertex density (WaveBlender-style) are the same representation at different resolutions. This dissolves MeshEditor's "excitable vertex subset" into a special case rather than a spec concept.

**Rejected.** Namespaced per-vertex mesh attributes (`KHR_audio_modal:SHAPE_n`, gaussian-splatting style) — one attribute per mode is unworkable for 30–100 modes, and it inherits every mesh-binding problem above.

## Evaluation: nearest-point baseline, barycentric opt-in via `indices`

**Decision.** φₙ(p) is the value at the nearest sample point; when `indices` supplies a triangulation over the sample points, implementations SHOULD instead project p to the closest point on that triangulation and blend the corner vectors barycentrically.

**Why.** Nearest-point is always defined, needs no topology, and is one kd-tree lookup — but it makes the field piecewise-constant over Voronoi cells, so sliding contacts step discretely between timbres (audible when sampling is sparse). Barycentric interpolation makes strike-position timbre vary continuously — exactly van den Doel's "sound map" (only gains interpolate; frequencies/decays are shared across positions), and what WaveBlender does onto FDTD boundary points. Interpolating the vectors is legitimate because mode shapes are smooth spatial fields (the smoothness Langlois 2014's compression exploits).

**Why SHOULD, not MUST.** Nearest-point converges to the same result as density grows; mandating closest-point-on-mesh machinery would raise the floor for minimal implementations without changing what densely sampled assets sound like.

**Known caveat (authoring-side).** A high-order mode whose shape flips sign between adjacent samples interpolates through zero along the edge, underestimating excitation — sample spacing must resolve the spatial wavelength of the highest mode that matters (same spirit as the FEM h < λ/6 rule). Degrades gracefully: low modes, which dominate perception, are smoothest.

## Per-mode decay rates required; Rayleigh material optional

**Decision.** `decayRates` (d, in s⁻¹, envelope e^(−dt)) is a required per-mode array. The acoustic material (ρ, E, ν, α, β) is optional metadata.

**Why.** Two provenance flavors must coexist: FEM-computed models (damping fully derivable from α, β — ModalSound doesn't even store it) and measured models (ACME, RealImpact — per-mode decays fitted from recordings, unconstrained by the Rayleigh circle). Baking d per mode is the only representation that covers both, and it gives renderers a single code path with no derivation branch. The material then serves three optional purposes: re-deriving damping after edits, uniform-scale rescaling (the β term is scale-dependent), and Hertz contact-duration estimation.

**Unit choice.** Decay rate d in s⁻¹ is the literature's canonical continuous-time form (van den Doel/Pai, JASS, Phya, RealImpact fits). The unit zoo (T60, Q, bandwidth, ξ, Rayleigh pairs) is the single biggest interchange hazard identified in the survey; the spec picks one and gives conversions in the appendix (T60 = ln 1000/d; Q = πf/d).

## Frequencies: observed (damped) Hz

**Decision.** `frequencies` stores the damped natural frequency in Hz — what a listener observes and what a resonator is tuned to.

**Why.** Hz is unambiguous and sample-rate-free. The trap being avoided is real: WaveBlender/openpbso store generalized eigenvalues λ = ω²ρ, requiring density to recover Hz — a convention a naive reader *will* get wrong. Damped (not undamped) frequency because measured models fit it directly and renderers consume it directly; the undamped ω is recoverable as √((2πf)² + d²) when needed (rescaling, damping re-derivation).

## Mass normalization convention

**Decision.** Shapes SHOULD be mass-normalized displacement eigenvectors (ΦᵀMΦ = I, M in kg), giving them units of kg^(−1/2).

**Why.** Density is baked into the shapes, so aₙ = φₙ·j carries no density term and a renderer produces correct relative levels from the shape data alone. This is what makes an optional `material` coherent, and it makes *relative* loudness between models physically meaningful (a heavy iron pot is quieter per unit impulse than a thin glass, because its larger mass matrix forces a smaller φ). The form also handles cases the alternative cannot: measured models with no mass matrix are fitted directly into it, and spatially varying density needs no special case because ρ lives inside M. The competing convention (Φᵀ(M/ρ)Φ = I, ρ factored out, used by openpbso/WaveBlender) keeps the eigensolve on a geometry-only mass matrix, but that is an authoring-time convenience irrelevant to a format that stores results, and it reintroduces ρ at render time (an explicit /ρ in the modal force) while assuming uniform density. The two are equivalent up to a √ρ factor on φ. Measured models cannot literally satisfy ΦᵀMΦ = I, hence SHOULD, plus the requirement that any data be *expressed in the same form* so the normative math produces intended relative amplitudes.

## Mode-major layout + importance ordering

**Decision.** `shapes` is mode-major (mode m's block at m·P + i); modes SHOULD be ordered by decreasing perceptual importance.

**Why.** Together these make "render only the first N modes" a prefix read of every accessor — free graceful degradation, the Phya/JASS precedent (both ship amplitude-sorted files). Contact-time lookup (all modes at one point) is a strided read either way. Importance ordering is SHOULD because FEM output is naturally frequency-ascending and reordering is a cheap authoring step, not a runtime necessity.

## Excitation semantics: impulse + position

**Decision.** An excitation is (j in N·s, p, t₀), both expressed node-locally: p by the full inverse global transform, j rotated only (physical magnitude preserved). Sustained contacts are impulse sequences, optionally shaped by the contact-force spectrum.

**Why.** Impulse + position is exactly the currency of `KHR_physics_rigid_bodies`' `rigid_body/applyPointImpulse` interactivity node — the one place the physics spec exposes "how hard and where." The physics spec deliberately does not standardize contact-event plumbing, so this spec matches: it defines what an excitation *is*, and leaves how the runtime obtains one out of scope (using the physics spec's own out-of-scope phrasing style). j is rotated but not scaled into local space because impulses are physical quantities; geometry scale effects are carried by the shape-rescaling rules instead. Contact-duration shaping is SHOULD-level and lightweight (Hertz τ derivation is possible from the material but deliberately not required — WaveBlender's version needs curvature and effective mass, far too heavy to mandate).

**Spectrum, not attenuation.** Scaling aₙ by F̂(fₙ) is the physical mechanism: a force of duration τ carries little spectral energy above roughly 1/(2τ), so high modes are never driven rather than filtered after the fact. Driving a resonator with the sampled pulse produces F̂(fₙ) with no filter to tune. An output low-pass reaches a similar result more coarsely, with a hand-tuned cutoff unrelated to contact mechanics.

**Temporal only.** A point impulse captures contact duration alone (the rolloff near 1/(2τ)). The finite contact *patch* (spatial averaging of φₙ, low-passing high spatial modes) and higher damping in soft materials (already in d) are out of scope, the former needing a contact radius the model does not carry.

## Synthesis: normative relative levels, free implementation

**Decision.** The normative output is the superposition Σ aₙ e^(−dₙt) sin(2πfₙt) with relative amplitudes across modes/excitations/instances normative and absolute level implementation-defined; any resonator with matching impulse response is admissible.

**Why.** This pins down what cross-implementation consistency actually requires (the *balance* of the sound) without dictating architecture — biquad banks, complex one-pole updates (MeshEditor), frequency-domain synthesis, and full wave solvers all qualify. Absolute level is the audio analog of exposure in rendering: platform/mixing territory. Sine phase is specified for definiteness but phase is perceptually irrelevant; the literature uses the same form.

## Acceleration noise in scope, mass properties self-contained

**Decision.** Rigid-body acceleration noise (the contact "click") is a SHOULD-level render feature driven by the same excitation as the modes, radiated omnidirectionally. Mass, center of mass, and inertia live in an optional `massProperties` block mirroring `KHR_physics_rigid_bodies`. When absent they MAY come from that physics extension or from watertight geometry plus ρ, with the model's own values authoritative.

**Why.** It is not a modal-core limitation but a cheap peer mechanism on the same impulse, and the primary sound for small stiff bodies whose modes are ultrasonic (which the old exclusion left silent). Its analytic form (Δv = j/M, Δω = I⁻¹(r×j), half-sine force pulse, dipole far field ∝ ρ₀V·ȧ) needs no precomputed radiation data. The radiated shape is the *derivative* of the acceleration, not the acceleration: a compact body recoiling without changing volume has no monopole, so its leading radiation is a dipole whose far-field pressure carries one extra time-derivative (Curle 1955, Morse & Ingard). That is what a wave solve radiates from the same motion, it drops the unphysical DC of the raw acceleration bump, and it is free (the derivative of a half-sine is a cosine lobe). Mass properties stay self-contained because an extension owns its data: a sibling as the primary source would drop the audio when physics is absent, and the model wins conflicts because acoustic mass and gameplay physics mass legitimately differ. Omnidirectional render matches the approximation the modal core already makes, directivity deferred like FFAT.

## Nyquist culling is a MUST

**Decision.** Implementations MUST NOT produce aliased output from modes at or above the output Nyquist frequency; equivalently, such modes contribute no output.

**Why MUST.** A resonator instantiated above Nyquist doesn't drop the mode — it aliases, folding to fs − f: a ghost partial with no physical interpretation. There is no legitimate rendering of such a mode at that output rate, so the requirement forbids exactly one thing (aliased garbage) and permits every real architecture: skipping (MeshEditor mutes modes at or above Nyquist), or high internal rates with anti-aliased decimation (WaveBlender), where the mode is equivalently absent from the output. As a MUST it guarantees the predictability that matters: an asset with modes to 20 kHz played through a 32 kHz output must sound like a low-passed version of itself, never gain inharmonic tones.

## Scale semantics

**Decision.** Modal data describes the object at its world size in the scene's *initial* state. Uniform rescale by γ: implementations SHOULD apply ω → ω/γ, φ → γ^(−3/2)φ, re-derive d from material α/β when present (the βω² term is scale-dependent), then f from (ω, d). Non-uniform scale: undefined.

**Why.** Modes are baked for one physical size — frequency scales inversely with size (bigger = lower pitch). The uniform laws are exact (Zheng & James 2010, Appendix E) and cheap, hence SHOULD; requiring them would burden minimal renderers, ignoring them (render unmodified) is the sanctioned fallback. Non-uniform scale changes mode *shapes*, not just frequencies — there is no recovery from precomputed data, so "undefined" (the `KHR_gaussian_splatting` precedent for unsupported transform regimes) is the honest answer. "Initial state" (pre-animation) is the anchoring a loader can actually detect; it also matches MeshEditor's practice of locking scale once a modal model exists.

## GPU instancing: one model instance per render instance

**Decision.** A node with `EXT_mesh_gpu_instancing` and a modal model instantiates the model once per render instance, each with independent oscillator state and the node's `gain`. Excitation attribution to a render instance is host logic (same scoping as contact reporting); once attributed, all excitation math uses the composed transform (node global × instance TRS), and each instance is a source at its own origin. The model is referenced to identity instance transform, so a per-instance uniform `SCALE` is handled by the standard scale-adjustment rules; non-uniform composed scale is undefined.

**Why.** Instancing is where modal audio pays off — piles of coins, debris, bricks — and striking one instance must ring only that instance. Treating instance `SCALE` as a uniform scale change turns per-instance size variation into physically correct detuning of a single shared model, which is exactly the runtime-rescaling soundbank technique of Zheng & James 2010. Attribution stays host-side because the physics layer doesn't define per-GPU-instance colliders either; the spec defines the semantics once a target instance is known, mirroring how contact plumbing is scoped.

## Out of scope, and why each exclusion is safe

Everything excluded *composes with* the modal core rather than replacing it:

- **Radiation transfer** — exactly separable: per-mode scalar amplitude field multiplying oscillator output (uncontested from PAT through KleinPAT). WaveBlender proves the same modal payload drives a full wave solve with no extra data. Future layer: per-mode FFAT-style cube maps (~2 KB/mode as 8-bit textures) map naturally onto glTF's image machinery.
- **Propagation/spatialization/listeners** — `KHR_audio_graph`/`KHR_audio_emitter`/platform territory; this extension only produces a monophonic source at the node.
- **Contact-event plumbing** — matches KHR_physics_rigid_bodies' own scoping.
- **Scrape/roll excitation synthesis** (surface profiles, fractal noise, speed-scaled filters) — a Phya-style *surface* descriptor is a coherent future sibling; the spec's impulse-sequence + low-pass language keeps the seam open.
- **Nonlinear/thin-shell coupling, fracture, sample hybrids** — known limitations of linear modal synthesis, documented as validity bounds (Authoring Notes) rather than half-specified features.
- **Analysis itself** — authoring-time; appendix documents the standard pipeline informatively.

**Anti-pattern deliberately avoided.** MPEG-4 Structured Audio standardized a programmable synthesis language and saw zero adoption; every surviving system in the survey ships declarative data with fixed resonator-bank semantics. This spec is strictly the latter.

## Optionality and fallback summary

- Extension is purely additive: `extensionsUsed` only; assets SHOULD NOT list it in `extensionsRequired`.
- Required per model: `frequencies`, `decayRates`, `positions`, `shapes`. Required per node object: `model`.
- Optional: `indices` (fallback: nearest-point), `material` (fallback: no rescaling/re-derivation/contact-duration estimation), `massProperties` (fallback: physics extension or watertight geometry + ρ), `gain` (default 1.0), `name`.
- Renderer latitude: mode-subset rendering MAY, barycentric interpolation SHOULD, uniform-scale adjustment SHOULD, duration shaping SHOULD, acceleration noise SHOULD; aliasing prohibition and linear superposition and per-instance state MUST.

## Deferred / future work

- Example + conformance test assets (deliberately after spec stabilization).
- Radiation-transfer layer (FFAT cube maps or multipole coefficients) as a sibling extension.
- Acoustic *surface* descriptor for sustained contact (roughness profile with physical spatial units, contact damping γ from Zheng & James 2011).
- Mode-shape compression (Langlois-style) if asset sizes demand it.
- `KHR_interactivity` excitation node (contact events carrying impulse + position) once that ecosystem settles.
- MeshEditor conformance: retain eigenvector 3-vectors in `mesh2modes`, import/export of the extension, direction-aware excitation.
