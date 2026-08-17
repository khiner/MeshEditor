# KHR_audio_rigid_bodies — Design Decisions and Rationale

Companion reference for the spec in `extensions/2.0/Khronos/KHR_audio_rigid_bodies/`. Not part of the specification. Records what the spec decides, why, and what would prompt revisiting.

Two bodies are covered here. Everything up to *Sustained contact is one excitation regime, not a second model*
is the settled core: modal models, acoustic materials, impact excitation, synthesis, acceleration noise and
radiation. Everything from it on is the sustained-contact half, which the specification gathers in one section
and which is still changing.

## Name and positioning: `KHR_audio_rigid_bodies`

**Decision.** KHR-prefixed name, Draft status, `extensions/2.0/Khronos/` layout — the glTF_Physics playbook (KHR-named draft staged for ratification, e.g. `KHR_physics_rigid_bodies`). The repository is `glTF_PhysicalAudio`, naming the domain rather than a technique.

**Why the scope.** Khronos naming convention is `<PREFIX>_<scope>_<feature>`; `audio` is the scope already established by the `KHR_audio_graph` / `KHR_audio_emitter` proposals. Naming into that scope signals the intended relationship: those proposals cover sources/emitters/listeners/spatialization; this extension is a complementary signal-*generating* layer they can consume. No modal or procedural-audio glTF proposal exists anywhere (verified across Khronos, OMI, MPEG, middleware), so this is a green field with no name collision.

**Why the feature is `rigid_bodies`.** It names the subject and the load-bearing assumption, which is what a reader needs in order to decide whether the extension applies. Naming a synthesis technique instead would under-describe the contents: acceleration noise and `massProperties` are rigid-body mechanisms the spec itself calls "distinct from the modal response," and the dominant sound for small stiff bodies whose modes are ultrasonic, while acoustic surfaces and contact force are not modal either. The name also makes the relationship to `KHR_physics_rigid_bodies` legible: same bodies, same contact events, one solving motion and the other the sound of small vibrations about it.

**Cost accepted.** Nobody searching "glTF modal sound" finds the name. `KHR_physics_rigid_bodies` has the same problem for "friction," and the Overview carries the keyword.

## One extension, not a family

**Decision.** Models, acoustic materials, acoustic surfaces, synthesis, acceleration noise, and sustained contact all live in `KHR_audio_rigid_bodies`. No data extension is broken out.

**Why.** The glTF_Physics split is narrower than it appears. `physicsMaterials`, `physicsJoints`, and `collisionFilters` are all root-level arrays *inside* `KHR_physics_rigid_bodies`. Only `KHR_implicit_shapes` is separate, and its stated justification is that shape data has multiple prospective consumers: a node with a shape need not be a rigid body. The test for a breakout is a *demonstrated consumer outside the extension*, not data-versus-behavior purity.

Acoustic materials and surfaces fail that test. Both are consumed only by contact sound, which lives here. Materials are in particular a rigid-body-audio concern rather than a contact concern: `alpha` and `beta` describe how a body rings and have nothing to do with contact at all.

A breakout would also have to survive a structural problem. The natural layering is properties of matter, then contact excitation, then response models, but the middle layer carries no JSON payload of its own: contact state, the composite-surface rule, and the force model are prose and an appendix. A layer with no schema is a section, not an extension.

**Revisit when** a second demonstrated consumer of surfaces or materials appears, most likely sample-based response models. `KHR_audio_materials` then splits out on the `KHR_implicit_shapes` precedent, and the node object grows nested roles at the same time. Restructuring at Draft is mechanical, so the option costs nothing to hold open.

## Node object: optional roles, no required reference

**Decision.** `modalModel`, `acousticSurface`, and `gain` are all optional on the node extension object. A node opts into whichever roles apply.

**Why.** Sustained contact is pairwise, and the second body is usually silent: a floor or table contributes its finish without ever sounding. Requiring `modalModel` would leave that body unable to describe itself. `node.KHR_physics_rigid_bodies` is exactly this shape, with `motion`, `collider`, `trigger`, and `joint` all optional and a node taking whichever it needs.

**Array names are prefixed** (`modalModels`, `acousticMaterials`, `acousticSurfaces`) following `physicsMaterials` / `collisionFilters`. Under an extension holding three kinds of thing, a bare `models` is ambiguous and a bare `materials` reads as core glTF materials at the reference site.

**Ratification strategy.** The realistic ladder is vendor prefix → EXT (second independent implementation) → KHR (Khronos vote). The KHR name is permitted pre-ratification when that intent exists (glTF_Physics precedent). Review Draft stage requires JSON schema + test assets + a third-party implementation; test assets are deliberately deferred until the spec settles.

## One extension, not a data/behavior split

**Decision.** A single extension carries both the data (models, materials) and the rendering semantics.

**Why.** glTF_Physics split `KHR_implicit_shapes` (data-only) from `KHR_physics_rigid_bodies` (behavior) because the shape data has multiple prospective consumers. Modal model data has exactly one consumer — a modal audio renderer — so a split would add ceremony without reuse. Revisit if a second consumer materializes (e.g. a haptics extension wanting the same modes).

## Root-level arrays + node index reference

**Decision.** `modalModels[]`, `acousticMaterials[]`, and `acousticSurfaces[]` live in a `KHR_audio_rigid_bodies` object on the glTF root (items inherit `glTFChildOfRootProperty`, so they get `name`); nodes instance a model and a surface by index with an optional per-instance `gain`.

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

Namespaced per-vertex mesh attributes (`KHR_audio_rigid_bodies:SHAPE_n`, gaussian-splatting style) do not work here: one attribute per mode is unworkable for 30–100 modes, and it inherits every mesh-binding problem above.

## Evaluation: nearest-point baseline, barycentric opt-in via `indices`

**Decision.** φₙ(p) is the value at the nearest sample point; when `indices` supplies a triangulation over the sample points, implementations SHOULD instead project p to the closest point on that triangulation and blend the corner vectors barycentrically.

**Why.** Nearest-point is always defined, needs no topology, and is one kd-tree lookup — but it makes the field piecewise-constant over Voronoi cells, so sliding contacts step discretely between timbres (audible when sampling is sparse). Barycentric interpolation makes strike-position timbre vary continuously — exactly van den Doel's "sound map" (only gains interpolate; frequencies/decays are shared across positions), and what WaveBlender does onto FDTD boundary points. Interpolating the vectors is legitimate because mode shapes are smooth spatial fields (the smoothness Langlois 2014's compression exploits).

**Why SHOULD, not MUST.** Nearest-point converges to the same result as density grows; mandating closest-point-on-mesh machinery would raise the floor for minimal implementations without changing what densely sampled assets sound like.

**Known caveat (authoring-side).** A high-order mode whose shape flips sign between adjacent samples interpolates through zero along the edge, underestimating excitation — sample spacing must resolve the spatial wavelength of the highest mode that matters (same spirit as the FEM h < λ/6 rule). Degrades gracefully: low modes, which dominate perception, are smoothest.

## Per-mode decay rates required; Rayleigh material optional

**Decision.** `decayRates` (d, in s⁻¹, envelope e^(−dt)) is a required per-mode array. The acoustic material (ρ, E, ν, α, β) is optional metadata.

**Why.** Two provenance flavors must coexist: FEM-computed models (damping fully derivable from α, β — ModalSound doesn't even store it) and measured models (ACME, RealImpact — per-mode decays fitted from recordings, unconstrained by the Rayleigh circle). Baking d per mode is the only representation that covers both, and it gives renderers a single code path with no derivation branch. The material then serves three optional purposes: re-deriving damping after edits, uniform-scale rescaling (the β term is scale-dependent), and Hertz contact-duration estimation.

**Why this survives better damping models.** Rayleigh damping is the special case β₁(A) = α₁A, β₂(A) = α₂A of generalized proportional damping (Adhikari 2006, applied to sound synthesis by Sterling & Lin 2016), and provably cannot fit a material whose damping-to-frequency exponent differs from Rayleigh's fixed one. That is a real representational limit. Whether it is an *audible* one is unsettled: Sterling & Lin's own listening test found their power-law GPD model rated less realistic than Rayleigh, and they read the outcome as validating Rayleigh rather than displacing it.

A per-mode array is indifferent to the whole question, representing any distribution exactly, so damping-model research lands at authoring time and changes nothing about the format. It is also where current work already sits: differentiable modal renderers learn a damping factor per mode from recordings rather than deriving it from a material (Jin et al. 2024), and their synthesis form is the one this spec already defines.

The only render-time use of the Rayleigh form is re-deriving damping after a uniform rescale, and that rule is phrased in terms of the material's damping function so a future general form needs no amendment. An author with fitted decay rates omits α and β, and the existing fallback leaves the measured values alone.

**Carrying a damping curve in the material was rejected.** An authoring tool fits the curve, evaluates it per mode, and bakes the result, so the interchange format never needs it. The mixed perceptual evidence above makes the case weaker still: it would be new authored data serving one optional rule, in exchange for a fidelity gain that has not been demonstrated to be audible.

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

**Decision.** The normative output is the superposition Σ 2πfₙ aₙ e^(−dₙt) sin(2πfₙt) with relative amplitudes across modes/excitations/instances normative and absolute level implementation-defined, and any resonator with matching impulse response is admissible.

**Why.** This pins down what cross-implementation consistency actually requires (the *balance* of the sound) without dictating architecture — biquad banks, complex one-pole updates (MeshEditor), frequency-domain synthesis, and full wave solvers all qualify. Absolute level is the audio analog of exposure in rendering: platform/mixing territory. Sine phase is specified for definiteness but phase is perceptually irrelevant; the literature uses the same form.

## The source signal is pressure, so the two mechanisms are one quantity

**Decision.** The source signal is the far-field pressure a compact source radiates. The modal sum therefore carries a factor of 2πfₙ, and per-mode acoustic transfer data replaces that factor rather than multiplying it. Referring the factor to a fixed frequency is permitted, being a uniform absolute scale.

**Why.** Without it the specification prescribed superposing two different physical quantities. A mode's impulse response aₙ e^(−dₙt) sin(2πfₙt) is its surface displacement times 2πfₙ, so it is a velocity, while acceleration noise was already defined as a far-field pressure with the dipole derivative baked in, and the text prescribes adding them. Their balance was then wrong by a factor of 2πfₙ, which both sections declare normative, and it mattered most where the specification says acceleration noise carries the sound: small stiff bodies whose modes are ultrasonic.

Pressure is the right common quantity rather than velocity. Radiation was already inside the specification for one of the two mechanisms, with a stated reason (a raw acceleration bump carries an unphysical DC that the dipole derivative removes), and the same reasoning applies to the modal term, whose 6 dB per octave tilt is the difference between a dull and a bright strike. A renderer with no transfer model still has to produce what a listener would hear. And the empirical laws the model is checked against, from the roughness-noise literature, are all microphone pressure, so an output in any other quantity cannot be compared to them.

Velocity remains the right interface quantity for a radiation solver, which is why mode shapes stay displacement eigenvectors: the Helmholtz boundary condition is normal velocity, and every transfer method takes it. The specification carries the compact-source factor so that a renderer without transfer data is correct at leading order, and names it explicitly so that a renderer with transfer data knows what to drop.

Measured against the roughness-noise laws on a sliding contact, the change moves the roughness exponent from 0.82 to 0.69 and the speed exponent from 0.46 to 0.68, which places both in the range direct numerical simulation of sliding rough surfaces reports (Dang et al. 2013) rather than one of the two.

## The level is pressure at a stated distance, and the listener term stops at 1/*d*

**Decision.** An implementation wanting a physically meaningful level states the quantity its output carries and the reference distance *r*₀ it carries it at, renders far-field pressure in pascals there, and attenuates each body by *r*₀/max(*d*, *r*₀). Audibility culling stays at the *r*₀ level.

**Why a stated quantity and distance.** Two assets, two objects, and two mechanisms are only comparable in one mix when all of them name the same quantity at the same distance. Once they do, each mode's amplitude follows from the air it moves, so nothing has to be balanced by a chosen gain.

**Why the inverse-distance term is in scope when propagation is not.** A stated output level is meaningless without the distance term that carries it, so this one term has to be here while directivity, occlusion, and reverberation stay out. Holding the level inside *r*₀ bounds the sample a listener passing through a body produces, which an unclamped 1/*d* does not.

**Why culling ignores the listener.** A body that dropped modes as the camera receded would change its own decay as the camera moved, so the live set is decided at the *r*₀ level and attenuation scales only the output.

## Acceleration noise in scope, mass properties self-contained

**Decision.** Rigid-body acceleration noise (the contact "click") is a SHOULD-level render feature driven by the same excitation as the modes, radiated omnidirectionally. Mass, center of mass, and inertia live in an optional `massProperties` block mirroring `KHR_physics_rigid_bodies`. When absent they MAY come from that physics extension or from watertight geometry plus ρ, with the model's own values authoritative.

**Why.** It is not a modal-core limitation but a cheap peer mechanism on the same impulse, and the primary sound for small stiff bodies whose modes are ultrasonic (which the old exclusion left silent). Its analytic form (Δv = j/M, Δω = I⁻¹(r×j), half-sine force pulse, dipole far field ∝ ρ₀V·ȧ) needs no precomputed radiation data. The radiated shape is the *derivative* of the acceleration, not the acceleration: a compact body recoiling without changing volume has no monopole, so its leading radiation is a dipole whose far-field pressure carries one extra time-derivative (Curle 1955, Morse & Ingard). That is what a wave solve radiates from the same motion, it drops the unphysical DC of the raw acceleration bump, and it is free (the derivative of a half-sine is a cosine lobe). Mass properties stay self-contained because an extension owns its data: a sibling as the primary source would drop the audio when physics is absent, and the model wins conflicts because acoustic mass and gameplay physics mass legitimately differ. Omnidirectional render matches the approximation the modal core already makes, directivity deferred like FFAT.

**Impulsive excitation only.** A draft carried the same mechanism over to sustained contact, with **F**(*t*) in place of the impulse pulse. That is derivable, since the dipole result assumes only a compact body under a net force and the net force on a loaded body is the fluctuation the excitation already is, but it is not how the field models sliding noise and it was withdrawn. Le Bot 2017 projects the vibration onto a modal basis driven by modal contact forces, with no rigid-body term, and his own simulation slides a steel cube on an elastic plate and radiates from the plate. A body in sustained contact is also coupled to the counterface through the contact stiffness rather than free, so its mass and that stiffness form a resonator, and radiating from a recoil the contact model does not see would be the wrong side of it. Whether the mechanism is negligible for sliding or merely unstudied is open; the spec does not assert it either way.

## Nyquist culling is a MUST

**Decision.** Implementations MUST NOT produce aliased output from modes at or above the output Nyquist frequency; equivalently, such modes contribute no output.

**Why MUST.** A resonator instantiated above Nyquist doesn't drop the mode — it aliases, folding to fs − f: a ghost partial with no physical interpretation. There is no legitimate rendering of such a mode at that output rate, so the requirement forbids exactly one thing (aliased garbage) and permits every real architecture: skipping (MeshEditor mutes modes at or above Nyquist), or high internal rates with anti-aliased decimation (WaveBlender), where the mode is equivalently absent from the output. As a MUST it guarantees the predictability that matters: an asset with modes to 20 kHz played through a 32 kHz output must sound like a low-passed version of itself, never gain inharmonic tones.

## Scale semantics

**Decision.** Modal data describes the object at its world size in the scene's *initial* state. Uniform rescale by γ: implementations SHOULD apply ω → ω/γ, φ → γ^(−3/2)φ, re-derive d from material α/β when present (the βω² term is scale-dependent), then f from (ω, d). Non-uniform scale: undefined.

**Why.** Modes are baked for one physical size — frequency scales inversely with size (bigger = lower pitch). The uniform laws are exact (Zheng & James 2010, Appendix E) and cheap, hence SHOULD; requiring them would burden minimal renderers, ignoring them (render unmodified) is the sanctioned fallback. Non-uniform scale changes mode *shapes*, not just frequencies — there is no recovery from precomputed data, so "undefined" (the `KHR_gaussian_splatting` precedent for unsupported transform regimes) is the honest answer. "Initial state" (pre-animation) is the anchoring a loader can actually detect; it also matches MeshEditor's practice of locking scale once a modal model exists.

## GPU instancing: one model instance per render instance

**Decision.** A node with `EXT_mesh_gpu_instancing` and a modal model instantiates the model once per render instance, each with independent oscillator state and the node's `gain`. Excitation attribution to a render instance is host logic (same scoping as contact reporting); once attributed, all excitation math uses the composed transform (node global × instance TRS), and each instance is a source at its own origin. The model is referenced to identity instance transform, so a per-instance uniform `SCALE` is handled by the standard scale-adjustment rules; non-uniform composed scale is undefined.

**Why.** Instancing is where modal audio pays off — piles of coins, debris, bricks — and striking one instance must ring only that instance. Treating instance `SCALE` as a uniform scale change turns per-instance size variation into physically correct detuning of a single shared model, which is exactly the runtime-rescaling soundbank technique of Zheng & James 2010. Attribution stays host-side because the physics layer doesn't define per-GPU-instance colliders either; the spec defines the semantics once a target instance is known, mirroring how contact plumbing is scoped.

## Sustained contact is one excitation regime, not a second model

**Decision.** Sliding, scraping, and rolling are rendered as a continuous force driving the same modes, mode shapes, and mass properties an impact drives. Rolling and sliding are not separate models.

**Why.** The physics distinguishes them by measurable speeds, not by a category. Each body has its own sweep speed, the rate the contact travels over that body's surface, and slip is the difference between the two. Pure rolling has both sweeps equal so slip vanishes; a box sliding on a fixed floor has zero sweep on the box and sweep equal to slip on the floor; partial slip lies between, and a body that rolls then skids moves continuously across that range.

**Sweep is per body, which is easy to get wrong.** Collapsing the two into a single sweep speed silences the canonical scraping case: the sliding box's own sweep is zero, since the same material region stays in contact, so a single-rate traversal never advances and the patch cutoff collapses to zero. Each surface must be traversed at its own rate and the contributions summed. A body sounds because the *other* surface streams past its stationary contact patch. A single synthesized track standing in for both can carry only one rate, so it follows the faster sweep and is exact only where the other surface is at rest relative to the contact, which is why the spec states when the fallback holds. Any implementation that switches between a "rolling model" and a "sliding model" introduces an audible transition at a place where the physics has none, which is why the spec says SHOULD NOT rather than leaving it open. Both speeds fall out of a rigid-body solver at no extra cost, so this needs no estimation heuristics.

**Implication for the excitation definition.** The impulse form is the limit of the sustained form as contact duration goes to zero, which the spec states. A renderer driving modes with a sampled force pulse (SHOULD-level, for contact-duration shaping) renders sustained contact by substituting a different force signal, with no other change.

## Contact force: constrain behavior, not formula

**Decision.** The spec states properties any conformant contact force must have (non-negative, silent at rest, traversal indexed by distance, √v·N amplitude, roughness low-passed by the region that carries the load, force limited far above the load) and puts one satisfying model in a non-normative appendix.

**Why the clamp is a MUST and the force law is not.** Perret-Liaudet and Rigaud 2003 solve one preloaded Hertzian contact under random drive, then replace Hertz's 3/2 power with a linear law carrying the same contact-loss clamp, and find the response identical around the primary peak and differing only well above it. The separation nonlinearity is what shapes the sound. The law between separations is a high-frequency correction. Constraining the clamp and leaving the exponent free therefore costs nothing in consistency between implementations and leaves the expensive part of the model open.

**Why the per-region clamp is a SHOULD.** A rough contact carries its load on discrete regions that release and re-engage at different moments, and that intermittency is the sound. A model clamping only the contact's aggregate force never separates at all under a load its mean can carry, so it renders a smooth force with the intermittency averaged out. That makes the guidance worth stating, but it is guidance about a decomposition rather than about the force two bodies exert, and an implementation is free to resolve the contact however it likes or not at all. The section's MUSTs are reserved for what shows in the output (non-negative force, equal and opposite excitation, silence at rest), and its SHOULDs for how a model reaches it, which is where this belongs.

**Why the reference bed stays in the appendix.** Rendering the contact as a bed of Hertz spots is one way to satisfy the clamp guidance, and Appendix B carries it as such. Naming it normatively would fix an implementation where the extension only needs an audible result, and would exclude models that reach the same behaviour by resolving the interface directly.

**The limit sits far above the load, not at it.** A sliding rough contact meets its counterface as a succession of micro-impacts whose peaks reach many times the mean load, which is the premise of the roughness-noise literature (Grégoire et al. 2021 measure the individual events). A limit at the load compresses that fluctuation rather than bounding it, and it is the roughness itself being compressed: sweeping the knee from one times the load to a hundred moves the measured roughness exponent from 0.26 to 0.82 against a literature range of 0.7 to 0.96. A limit is still needed, because both force laws grow without bound in approach and a contact whose geometry changes under it can report an approach from far outside the regime either law describes.

**Why.** This matches how [Synthesis](#synthesis-normative-relative-levels-free-implementation) is already scoped: pin the perceptible behavior, leave the architecture free. Mandating Hunt and Crossley would exclude equally valid formulations, and mandating nothing would let a renderer emit fixed-rate noise that neither tracks speed nor stops, which is qualitatively wrong rather than merely different.

**Two requirements are MUST**, on the same reasoning as the Nyquist rule: each forbids exactly one audible failure and permits every real architecture. Non-negative force forbids a contact that pulls, and preserves the separation nonlinearity that produces micro-collisions and chatter. Silence at rest forbids a settled body humming forever, which is the single worst failure mode for this feature and the one a naive implementation falls into by leaving the equilibrium Hertz force in the excitation.

**Parametric surfaces are statistical, stored profiles are reproducible.** A surface given only by roughness, correlation length, and spectral slope specifies an ensemble, so two renderers agree on character but not sample-for-sample. The optional `profile` accessor is the escape hatch, and it also matches the format's philosophy of shipping baked results rather than standardizing a generator (the MPEG-4 Structured Audio anti-pattern noted below). Measured tracks are small: a few thousand samples at a few microns covers enough surface to loop inaudibly.

**The spectral slope is the one-dimensional exponent, and the fractal relation is the self-affine one.** *p* is the exponent of a profile rather than of an areal map, matching what `profile` carries and what a contact traverses. For a self-affine surface the two-dimensional power spectrum goes as *q*^−2−2*H* and a line scan of it as *q*^−1−2*H*, one power less steep (Jacobs et al. 2017 Eq. 48, derived there from the two-dimensional form). Inverting the one-dimensional form gives the Hurst exponent *H* = −(1 + *p*)/2. The surface's fractal dimension is 3 − *H* (Persson et al. 2005 Appendix B derives it by box counting) and a planar section drops that by one, so the profile's is 2 − *H* = *p*/2 + 2.5. *H* is the quantity the contact-mechanics literature states results in, and the profile *D* is the quantity surface metrology reports, so the spec gives both.

van den Doel et al. 2001 state *D* = *p*/2 + 2 for the same profile quantity, following a phonograph-needle model whose wavetable is a height track exactly as `profile` is. That form is half a unit off, and following it puts the presets in the wrong place rather than merely renaming them: the machined row's measured *D* = 1.3 becomes *p* = −1.4 and so *H* = 0.2, against the 0.7 to 0.9 that spectral analysis reports for surfaces from atomic to geological scales (Jacobs et al. 2017) and the 0.8 that contact-mechanics work takes as the common case while calling lower values rare (Papangelo et al. 2017). The two forms are easiest to separate at *p* = −2, where the self-affine relation gives the Brownian value *D* = 1.5 and van den Doel's gives *D* = 1, a smooth rectifiable line, which a nowhere-differentiable Brownian profile is not. The measured fractal dimensions are the anchor, so the table keeps its *D* column and recomputes *p* = 2*D* − 5.

**The *D* column's spread is wider than measurement supports.** Only the machined row is anchored. Recomputing *p* from the other three carries their illustrative *D* values through to *H* = 0.9 for polished, 0.6 for sandblasted, and 0.5 for cast, and the last two sit below the band the literature reports. Closing that gap means measured fractal dimensions per finish, which these rows do not claim to be. What separates the finishes here is σ and *ℓ*, which span three orders of magnitude across the table.

## One real contact area across every regime, not a case split

**Decision.** The area the asperities touch over comes from a single expression, *A* = *A*₀ erf(√π *N*/(2 *A*₀ *p*)) with *p* = *E*\*|∇*h*|<sub>rms</sub>/2 and *A*₀ the smaller of the Hertz patch and the shared polygon (Pastewka and Robbins 2016). The microscale finish is filtered over the width of *A* and longer relief over the width of *A*₀. The force law still distinguishes a load-set contact from a geometry-set one, and the area does not.

**Why one expression rather than two cases.** Its limits are the two cases. At light load the error function linearizes to *A* = *N*/*p*, independent of *A*₀, which is the nominally flat rough result. At heavy load it saturates at *A*₀, which is the Hertz result. Between them it interpolates with no free parameter and no threshold, and it was validated against molecular simulations of rough spheres from 30 nm to 30 µm over ten orders of magnitude in load. A case split on geometry would have to answer a question the geometry cannot: the physical crossover is set by *h*<sub>rms</sub>/δ, the roughness over the smooth-surface penetration, which is Hertz-like below 0.01 and roughness-dominated above 10 (Tiwari and Persson 2020). Ordinary objects span that whole range, so a curved contact on a rough surface sits in the middle while a geometric test would still send it to Hertz.

**Why the finish and the relief are filtered over different widths.** The two widths differ by orders of magnitude, so one of them cannot serve both scales. Filtering the microscale finish over the confining region instead attenuates it to nothing wherever two faces meet, since that region is the whole shared polygon while the finish's own contacts span microns.

**Why not Greenwood and Tripp.** Theirs is the classical treatment of the rough sphere and it is superseded. It builds on the Greenwood and Williamson model of noninteracting asperities of identical radius, whose limitations for total contact area and for contact geometry are what motivated the work above. The same asperity-model family is the one the community's contact-mechanics comparison finds unreliable for interfacial separation and for the distribution of contact patch sizes (Müser et al. 2017), which are the quantities a contact sound model consumes.

**The asperity pressure is not a fitted constant.** *p* = *E*\*|∇*h*|<sub>rms</sub>/2 carries neither the load nor the area, which is what makes the real area proportional to load at light load. Persson's theory gives it analytically and the numerical studies Pastewka and Robbins draw on give the same coefficient to within the accuracy either claims.

**The force law is no longer branched either.** Two faces bear on the asperities standing proud inside the region confining them, so a geometry-set contact is a bed of Hertz spots and each spot takes the load-set law at its own radius. Persson's *N* exp(δ/*u*₀) is the *mean* of such a bed, and a mean is the wrong object here: the sound is made by individual spots parting and re-engaging, which averaging removes. The exponential's separation scale *u*₀ survives as the scale the bed's mean force decays over and as a check on its stiffness.

## The asperity radius is read at the surface's finest scale

**Decision.** Where an implementation renders the contact as a bed of discrete Hertz spots, each spot's radius of curvature is the one the surface carries at the shortest wavelength it resolves, ρ = 2⟨|∇²*h*|²⟩^(−1/2) (Pastewka and Robbins 2016). It is not coarsened to the width of the spot bearing on it.

**Why.** That radius is the one validated against molecular simulation in the low-load limit, where contact is dominated by the first asperity to touch. It also reconciles the bed with [the area law](#one-real-contact-area-across-every-regime-not-a-case-split): summing the areas of Hertz spots over the height distribution reproduces the erf law's area to within a few percent, and the two are independent accounts of one quantity. Coarsening the radius to the spot width breaks that agreement, because a spot's area grows with its radius while the area law knows nothing about curvature at all.

**Why the tempting alternative is wrong.** Müser et al. 2017 observe that most contact points belong to patches far larger than the finest scale, while rms curvature is set by the finest scale, and conclude that one "may not treat the asperity with the radius of curvature as measured on top of the asperity at the finest scale." Read alone that argues for a blunter radius. It does not: the same comparison faults bearing models for overestimating typical contact patch size by a factor of ten, and a blunter radius makes each spot larger still. The observation is that a real contact patch is a merged, ramified region formed by long-range elasticity between peaks, so the single-radius asperity picture does not describe its geometry. That is a limit on what the model class can represent rather than a correction factor inside it.

**Two conventions for the same number.** Their equation states ρ both as 2⟨|∇²*h*|²⟩^(−1/2), over the two-dimensional Laplacian, and as λ<sub>s</sub>/(2π|∇*h*|<sub>rms</sub>) √(2(2−*H*)/(1−*H*)) in surface parameters. The factor of two in the first is spent on the conversion between a profile's second derivative and the Laplacian, so an implementation measuring curvature along a one-dimensional trace matches ρ without applying it. The second form is the one to check against, since it needs no moment conversion. The two agree over the range a surface holds its self-affine form in and part as *H* approaches one, where √(1/(1−*H*)) diverges while a sampled second difference holds at the sampling scale.

**Revisit when** the bed carries a patch-size distribution rather than a population of equal spots. The radius then describes a patch rather than a summit, and the scale it is read at follows.

## The element ribbon is sized in metres, and its strip count carries the load

**Decision.** The elements a sustained contact bears on tile a ribbon of the interface whose extents are lengths rather than sample counts. Along the sweep the ribbon runs the distance a slide covers before the surface repeats, and across the sweep it samples one strip four correlation lengths wide. The width a contact actually bears over is that ribbon's area divided by its length, and the springs carry that width as the count of strips it holds.

**Why.** A sample count standing in for a length makes the physics track the sampling. The field is synthesized at the spacing the surface's own short-wavelength cutoff sets, so a fixed count of columns and rows covers less ground every time the band is described more finely: the same contact then bears over less area, seats deeper for it, and reports every bearing statistic at a contact the pair is not making. Pastewka and Robbins 2016 state their own domain the same way this decision does, as a band between two wavelengths plus a padding region that keeps the periodic images apart, with the discretization following.

**Why the strip is four correlation lengths, and why it is not the whole bearing width.** Resolving a surface's finest wavelength across a footprint tens of millimetres wide is four orders of magnitude of samples per axis, so the across-sweep direction is sampled rather than covered. What the strip must be wide enough for is the surface's own statistics, and Pastewka et al. 2013 Eq. (B3) bounds that: a domain of *L* against a spectral corner at *l* holds a fraction (1 + *H*(1 − (*l*/*L*)²)) / (1 + *H*) of the surface's height variance, which at *L* = 4*l* and *H* = 0.7 is 0.987. Past that a wider strip buys a larger bearing sample and nothing else, which is what `SPRING_STRIP_CORRELATIONS` brackets.

**What it is worth, measured.** On PressedRing/b_Grip the ribbon holds 272 strips, so the load per strip falls by that factor and the contact seats at its own pressure for the first time. The tilt then reads 0.512 against 0.403, the flank spring 7.9e6 N/m against 1.6e6, and the seated bench's top rung moves from *n* = 2.49 to 2.59, inside the measured 2.5 to 3 microslip band the law previously only reached the edge of. Across a fourfold refinement of the band the field's own statistics hold: summit density 2.28, 2.56, 2.64 × 10¹⁰ per m², mean summit stiffness 4.13, 4.00, 3.97 × 10⁷, mean element crest 7.52, 7.62, 7.61 μm, and the strip and bearing widths exactly flat where a fixed row count narrowed the strip fourfold.

**What is left is the ribbon's own asperity statistics rather than the element crest.** Four correlation lengths is enough for the strip's *spectrum* and far too little for its *bearing population*. Pastewka & Robbins 2016 Eq. (5), with the sphere radius replaced by the asperity radius, gives the load below which contact is carried by the first asperity rather than by a statistical population: Nc = 0.092 N for this pair. The whole contact carries 4.90 N, 53 times that, and is comfortably multiasperity, but one sampled ribbon at four correlation lengths carries 4.90/272 = 0.018 N, a fifth of Nc. Their own rule is that "linear scaling requires a statistical number of asperities," and a ribbon that narrow has none.

Measured across four strip widths at three surface realizations each, the flank moment falls steeply while the ribbon is below Nc and goes flat once it is above: 16.9, 6.23, 4.03, 4.18 × 10¹³ at 4, 16, 32 and 64 correlation lengths, whose ribbon loads are 0.20, 0.78, 1.56 and 3.1 times Nc. The mean bearing width flattens with it, 7.12, 9.53, 10.26, 10.24 × 10⁻⁷ m. So Eq. (5) predicts the threshold at twenty correlation lengths and convergence is observed at thirty-two.

**So a ribbon gathers several strips rather than widening one.** Reaching the threshold by widening a single strip costs memory as its square, and at lighter loads it is unreachable outright: N/Nc ≈ 10 would want an eleven-millimetre strip and a multi-gigabyte field. Instead the ribbon synthesizes eight independent realizations of the same surface one at a time, gathering each one's summits into the same elements, and takes each element's crest as the tallest point it reaches over all of them. The elements of one ribbon stand side by side across the sweep, so that crest is what the body meets first, and the population they share is what carries the load. Only the summits persist between realizations, a few megabytes against the field's hundreds.

**What it is worth, measured.** Eight gathered strips of four correlation lengths reproduce one coherent strip of thirty-two: flank moment 3.67 × 10¹³ against 4.03 for the coherent thirty-two and 4.18 for the coherent sixty-four, all three far from the single strip's 16.9, at three surface realizations each and a peak memory of 131 MB against 1.05 GB. `SPRING_STRIP_REALIZATIONS` counts them and `SPRING_STRIP_CORRELATIONS` sizes one, and their product is the ribbon.

**Revisit when** a contact's transverse extent is small enough to hold fewer strips than the ribbon gathers, where the ribbon would be sampling more interface than the contact has. The strip count already floors at one, so such a contact scales down rather than up, but nothing yet checks that the gathered ribbon fits inside the footprint it stands for.

## A measured profile reaches the elements by inverting its own spectrum across the sweep

**Decision.** A surface carrying a measured height track builds its element springs from a field that track is a row of. The field's radial spectrum is read from the track's own by inverting the integral a cut takes over the perpendicular wavenumber, the track's heights land on the field exactly by conditioning the draw on them, and the field spans the length the track was measured over at the sampling it was measured at. So a measured surface repeats where its measurement ends rather than at the synthesized ribbon's own sweep budget, and every other band the pair carries is drawn on that same grid. The correlation length and spectral slope such a surface needs are fitted to the track rather than read from the parameters beside it.

**Why.** The elements tile a two-dimensional field and a measurement is one trace, so the two do not meet without a construction. Leaving them apart is not a limit a measured surface has to live with. It is a gap: a surface carrying a track silently keeps the statistical bed however the container is flagged, which makes the measured arm of a measured-versus-synthesized comparison a bed arm. It also blocks the bed's deletion outright, since the bed would be the only path a measured surface has.

**Why the spectrum rather than the three parameters fitted to it.** A track is the one place in this model where a real surface enters, and its spectrum is what a contact bears on. Fitting it to a corner and a slope first and synthesizing from those would keep the *authoring* route and discard the measurement, which is the whole reason a profile is stored. The fitted band is still needed, because an element's width is a share of a correlation length and the roughness standing inside a bearing contact continues the slope below the band the measurement resolves, and neither of those is readable off the heights directly. Those two numbers are fitted to the track's own spectrum, so a surface that ships a profile needs no parameters beside it.

**Why the inversion is regularized rather than solved exactly.** A cut of an isotropic field integrates the radial spectrum over the perpendicular wavenumber, and on a grid that integral is triangular: every term of a cut at one wavenumber stands at a radial wavenumber at or above it, so the radial amplitudes solve from the top of the band down. Solving them that way is unusable. At a wavenumber *q* the cut's own bin holds only about 2·d*q*<sub>y</sub>/(π *q*) of what the cut carries, which on a strip one correlation length wide is one part in thirteen at the middle of a machined finish's band and one part in twenty-five near its top, so the exact solve amplifies the trace's own scatter by the inverse of that share and returns a spectrum full of holes, whose one-sided clamping then hands the field more power than the trace has. Measured on that strip, the miss ran 44 to 96 percent across the decaying band. The inversion is ill posed in the same way Abel's is and needs a prior, so what is inverted here is the smooth shape, on cells holding a sixteenth of an octave each, seeded from the same one-power-shallower relation the parametric path applies in closed form and refined multiplicatively against the sum a cut of the field actually takes.

**What it is worth, measured.** Over the decaying band the field's cuts carry the trace's own spectrum to within 1 percent, and their slope reads 1.006 of the trace's own. The measured heights come back off the field's row exactly, to 1e-4 of one root-mean-square height. **The parametric patch over the same surface and grid reads 24 to 37 percent low across that same band and 14 percent shallow in slope, and carries the plateau 22 to 35 percent high**, from two causes it never checks for. The perpendicular sum a cut takes is discrete, and a strip whose rows step by a sizeable fraction of the corner over-counts it, which lifts the plateau. And the band's cutoff is applied to the *radial* wavenumber, so a cut near the top of the band integrates a truncated arc and rolls off before the stated cutoff, where a track of the same surface carries the power law right up to it. The unit-root-mean-square normalization then pays for the plateau out of the rest of the band. The second of those is a convention question rather than an error, since a profile of a radially band-limited surface genuinely does roll off early, but the track and the patch currently answer it differently while the model reads both as one surface. Both arms carry the plateau high at the low end, which is the strip being four correlation lengths wide and its perpendicular grid therefore starting at a quarter of the corner.

**Revisit when** the parametric path's own cuts are worth correcting. The refinement above is written against a measured spectrum, and the same refinement against a parametric one would close the 24 to 37 percent, at the cost of an inversion in a path that currently evaluates a closed form.

## The oblique-flank moment is summed over bearing contacts, not over elements

**Decision.** The moment that sizes oblique-flank micro-slip is Σ slope²/force over the summits carrying load, tabulated against engagement alongside each element's force curve, and the width the flank tilt is read at divides the bearing stiffness by that same count of summits. An element is not a contact.

**Why.** The loop area a tilted flank sheds per cycle is (4/3)*k*<sub>t</sub>²(*a* sin θ)³/(μ*F*) for one contact, so the population's moment is a sum over contacts. An element of a quarter correlation length holds tens of summits and several of them can bear at once, and (Σ slope)² / (Σ force) equals Σ slope²/force only where every bearing summit sits at one depth. Otherwise it is smaller, by Cauchy-Schwarz, and the gap grows with the spread of depths the element holds, which makes the moment a reading of the tiling. The same argument applies to the width: d*F*/d*d* = 2*aE** holds for one axisymmetric contact, so dividing an element's aggregate stiffness by the count of elements measures the element rather than the contact.

**What it is worth, measured.** Where a contact is light enough that one summit bears per bearing element the two forms agree exactly, which is the case on PressedRing at its own pressure for one surface realization in four. Across four realizations the moment rises by up to 1.5x. On the seated bench, which presses hard enough for several summits per element, the top rung's amplitude exponent moves from *n* = 2.59 to 2.71 at a flank share of 76%, against a measured microslip band of 2.5 to 3.

## The flank tilt is read at the bearing patch's width, where the radius is not

**Decision.** The tilt that oblique-flank micro-slip slides against is the surface's rms gradient low-passed at the width of the contact its bearing asperities carry the load on, 2*a* for *a* = √(*Rd*) at the Hertz depth *d* = 1.5*N*/*K*. It is not the gradient at the shortest wavelength the surface resolves.

**Why.** Pastewka et al. 2013 Appendix B splits a contact's own shape from the roughness inside it at *q* = π/*r*<sub>0</sub> for a contact of radius *r*<sub>0</sub>: their Eq. (B6) builds the mesoasperity's curvature from components below that wavevector and their Eq. (B18) leaves everything above it as roughness within the contact. A wavelength shorter than the patch does not tilt the patch. It is relief the patch sits across, and it forms its own contacts one level down the hierarchy, which a model carrying a single level of elements does not represent. Müser et al. 2017 section 4.2 reaches the same limit from the measured side, that the finest scale sets local quantities such as rms gradient while real contact points belong to patches far larger.

**Why this is not the blunter radius the previous entry refuses.** The two quantities answer different questions and take their scales from different places. The radius is a curvature that says how one summit's contact grows as it is pressed, and it is validated against molecular simulation in the first-asperity limit where the contact is smaller than the feature carrying it. The tilt is the orientation of the material plane that summit already slides on, so it is an average over the patch by construction and the patch's own width is what bounds it. Blunting the radius would inflate the contact area the bed and the area law agree on. Reading the tilt at the patch changes no area at all.

**What it is worth, measured.** The width is now the elements' own, from d*F*/d*d* = 2*aE**, and the surface states its short wavelength, so both the tilt and the scale it is read at are physical. Once the ribbon carries the contact's real pressure the bearing contacts are finer than the track's own spacing on PressedRing/b_Grip, so the tilt reads its finest-scale value there and the split costs nothing at that load. It matters where a contact bears harder: at the previous pressure, 272 times higher, the same tilt read 0.41 against a finest-scale 0.51.

**And a wider band cannot supply it.** Combining Hertz at one summit, the linear area law, and Pastewka & Robbins Eq. (6) for the summit radius, the load and the surface gradient both cancel and leave the bearing contact's width as 2*a* = (3/16)√(2(2−*H*)/(1−*H*)) λ<sub>s</sub>, which is 1.10 λ<sub>s</sub> at *H* = 0.7. The bearing contact sits at about one short-wavelength cutoff whatever the load, so authoring a finer band sharpens the summits at exactly the rate that shrinks their contacts. No authored band ever contains its own contacts.

Nor is a finer band authorable. ISO 3274's short-wavelength cutoff is a property of the measuring instrument, set by the stylus tip radius, rather than of the surface, and ISO 4288 puts a machined finish of Ra 0.1 to 2 μm on a 0.8 mm profile filter with a 2 μm tip, which resolves nothing finer than a few microns (mitutoyo_surface_roughness and willrich_surface_parameters, held in papers_surface_metrology.tar.gz, restate those tables rather than being the standard). So a surface authored from a measurement carries no roughness at the scale its own contacts form at.

## The roughness inside a bearing contact is a term, not a resolution

**Decision.** Every summit's depth divides between its own Hertz contact and the band below the surface's stated short wavelength, which stands inside that contact. The contact takes *x*, and the interface inside a contact of width *w* = 1.5*k*√*x*/*E** takes (3/*H*) γ *A* *w*<sup>*H*</sup>, where γ *A* *w*<sup>*H*</sup> is the mean separation that band holds open. The two compliances add in series, per summit, and the sum over summits is the stack.

**Why.** The previous entry proves a band can never contain its own contacts, so the level below is not reachable by describing the surface more finely and has to be carried as a term. Pastewka et al. 2013 Appendix B gives that term as a second elastic energy of the same power in load as the contacts' own, combining with the first as 1/θ = 1/θ<sub>0</sub> + 1/θ<sub>1</sub> (their Eq. B27), and its prefactor is γ, β and *H* alone. Inside the patch the interface follows Persson's law at the patch's own pressure, so its stiffness is the load over the separation, and integrating d*F*/d(depth) = *F*/*u* along the load leaves (3/*H*) *u* of depth. Both depths are then powers of the load, which is why the whole population shares one curve and a summit enters it through one scale of its own.

**The incremental law is the primitive, not the energy.** Appendix B writes *U*<sup>(1)</sup> = *u*<sub>1</sub>*F* as an evaluation at the load, not as the work integral of a separation that varies along the path. Taking it as the integral instead gives (3+*H*)/*H* rather than (3/*H*), a fifth too much depth. Persson's own K = *p*/*u*<sub>0</sub>, which the same appendix restates as its Eq. (B28), is the statement that holds pointwise.

**It cannot be aggregated.** These springs sit in series with each summit while the summits sit in parallel, so the stack is Σ<sub>i</sub>(1/*k*<sub>i</sub> + *u*<sub>i</sub>/*F*<sub>i</sub>)<sup>−1</sup> and no single series element at the anchor reproduces it. The separation also stops widening at the stated cutoff: a contact wider than that has the field's own resolved summits standing inside it, and reading the band at its full width would count them twice.

**What it is worth, measured.** The observable is band robustness, since the internals of a rough contact are resolution-dependent by construction and the rendered force is not. On PressedRing/b_Grip at 4.90 N, over a fourfold span of the authored short wavelength, the interface's own compliance at fixed load drifts 23% with the term against 43% without it. At the shipping band the bearing contacts narrow from 0.99 to 0.68 μm and nearly twice as many bear, while the interface's stiffness barely moves: the added compliance is paid for by recruitment.

**What is left is the shape half of the same appendix.** Its Eq. (B6) evaluates a contact's radius with the spectrum integrated up to π/*r*<sub>0</sub>, the contact's own width, where the implementation reads the curvature of the band as authored. The stated cutoff therefore still sets how sharp a summit is, which is the residual drift above. Closing it is a self-consistent solve, since the width depends on the radius and the radius on the width, and it must take the contact's width as its cutoff rather than any sampling: the moment ∫*q*<sup>5</sup>*C*(*q*)d*q* is dominated by its upper limit for *H* < 1, so only a physical length converges it.

## One contact stiffness, capped by the shared polygon

**Decision.** *k* = 2*E*\* *a* with *a* = min(√(*R*\*δ), √(*A*₀<sub>poly</sub>/π)), and the contact time is the collision against that one law, integrated. Hertz's τ = 2.868 (*m*\*²/(*E*\*² *R*\* *v*))^(1/5) and the flat punch's τ = π √(*m*\*/*k*) are its limits, not its cases.

**Why this is a unification rather than an interpolation.** 2*E*\* *a* is exact at both ends. Differentiating Hertz's own *f*<sub>n</sub> = (4/3) *E*\*√*R*\* δ^(3/2) gives d*f*<sub>n</sub>/dδ = 2*E*\*√(*R*\*δ), which is 2*E*\* *a* at the Hertz patch radius, and a flat punch of radius *a* on an elastic half space has stiffness 2*E*\* *a* outright. The two agree in value at the depth the patch fills, so the capped form is continuous with no constant fitted between them. The speed dependence then follows rather than being asserted: a power-law spring *f* ∝ δ^p gives τ ∝ *v*^((p−1)/(p+1)), which is *v*^(−1/5) while the patch grows and nothing once it has filled.

**And it removes a duplicated patch.** π*a*(δ)² is exactly the bounding region *A*₀ that [the real contact area](#one-real-contact-area-across-every-regime-not-a-case-split) already takes as the smaller of the Hertz patch and the shared polygon. The stiffness law and the area law now read one patch instead of each deriving its own.

**Why not branch on whether the physics reports an area.** That question is answered by the solver, not by the contact. Whether a manifold comes back with three points or two depends on tessellation and on how exactly the bodies met, so a body landing a degree off level would cross the branch and its contact time would step. The cap crosses smoothly in both directions: a polygon wider than the patch ever grows changes nothing, and one narrower takes over as soon as the patch reaches it.

**What remains inaccurate.** The punch is a rigid one on a smooth half space, the stiffest the pair can be. Real faces meet asperity first and the load rises as the roughness flattens, so the true contact is softer and longer than this, by whatever the interfacial stiffness adds in series with this bulk one. A contact with neither curvature nor a shared polygon is an edge or a corner, whose curvature is singular rather than zero, and it is still decided by the curvature floor and still unmodelled.

## One contact description, not a model per event

**Decision.** A contact is described once, by its geometry, load, materials, and roughness, and the force follows from that description in every regime. The contact zone resolves into springs distributed across it, each bearing only where the surface reaches it, so no branch selects between a curved-contact model and a flat-contact model, and no rule decides when a contact stops being an impact and starts being a scrape.

**Why not a model per event, which is what the field does.** Impact and scraping are usually given separate force models: smooth Hertz for the strike, a fractal roughness track for the slide. That works because the two are different excitations of a shared resonator rather than competing theories, and it has produced decades of convincing results. It fails us for three reasons. A format has to make the switching rules normative or renderers diverge audibly on identical assets, where a contact-mechanics description is renderer-neutral by construction. Rolling is neither event, and the models that switch need a third bespoke term for it, presuming a sphere or fitting constants, where one description covers it with the geometric term and the patch-filtered relief already present. And [vibrational coupling](#vibrational-coupling)'s contact-dependent damping needs the area currently in contact at every instant, which a model that exists only between events cannot supply, so switching would forbid a body ringing differently at rest than in flight.

**Why discrete springs rather than a filtered average.** Roughness shorter than the contact zone excites less than longer roughness does, an effect rolling-noise work calls the contact filter. Applying it as a spatial average over the zone removes too much: Thompson et al. 2003 find the analytical filter gives "too great an attenuation at high frequencies" against a distributed spring model driven by measured roughness. Springs that each bear only where the surface reaches them reproduce the average where all of them touch and a single following contact where one does, with the load deciding between those and nothing authored. The same evaluation yields the separation nonlinearity, because a spring that loses contact simply stops bearing, and the area currently in contact, which is what [vibrational coupling](#vibrational-coupling)'s damping needs.

**Which limit a contact sits in is a number, not a choice.** Hertz idealizes roughness away and Persson idealizes curvature away, so each diverges exactly where the other applies, at *R*\* → ∞ and at *A*₀ → 0. Two ratios decide which applies: *h*<sub>rms</sub> over the smooth-surface penetration, and the count of asperities bearing load. Both follow from quantities a contact already has, and across ordinary objects both span several decades, so a renderer that commits to one regime is wrong somewhere unremarkable.

**What remains inaccurate.** Springs that bear independently neglect the elastic coupling between them, which contact-mechanics work holds to matter even at light load, and the models that restore it solve a coupled problem per step that no realtime budget affords. Asperities are taken to deform elastically, which holds for hard ceramics and soft polymers and fails for machined steel, where the pressure at an asperity exceeds the hardness. A one-dimensional profile stands in for a two-dimensional surface, so the set of asperities bearing at any instant is drawn from a line rather than an area. Each is a place the contact is not yet described correctly, and closing them is what makes the result more faithful, not a reason to describe the contact more coarsely.

## Contact curvature comes from the object's own geometry

**Decision.** κ is the mean surface curvature at the contact point, read from the mesh of the collider the contact landed on, or from that of its nearest ancestor with a mesh. The extension carries no curvature data, so an implementation derives it.

**Why not the collision geometry.** A collider is a proxy chosen for the solver's convenience, and the usual choices erase exactly the quantity being asked for. A convex hull replaces curvature with facets, a primitive replaces it with an analytic shape that was never fitted to the surface, and both read a curved body as flat. Flat understates κ without bound, and since *R*\* = 1/(κ₁ + κ₂), *k* ∝ √*R*\* and *a* ∝ *R*\*^(1/3), the error propagates into the contact stiffness, the patch radius, and the contact time together. A ceramic bowl colliding as a hull would ring with the contact time of a flat plate.

**Where the two coincide.** A body whose collision shape is its authored shape has one geometry, and its analytic curvature is exact. The distinction bites only when a collider stands in for something finer.

**Why geometry resolves on its own walk.** Curvature is a fact about the shape under the contact, while an acoustic surface is authored finish. Resolving κ through the surface walk would make it depend on where an author chose to attach a finish, so a compound whose foot carries a mesh but no surface would read the curvature of the body's mesh instead of the foot's. The two walks look alike and answer different questions, so they are stated separately.

**Cost accepted.** Deriving κ at the contact means locating the contact on the body's surface geometry, which a renderer holding only collision proxies cannot do. That is the same capability the mesoscale relief needs to sample `normalTexture` at **p**, so the two stand or fall together.

**Mean curvature is the axisymmetric form.** Johnson's general solution writes the gap between two surfaces as *Ax*² + *By*², where *A* + *B* is half the sum of all four principal curvatures and equals κ₁ + κ₂ for mean curvatures. Where the contact is axisymmetric the patch is a circle and that sum is the whole solution. Where the principal curvatures differ the patch is an ellipse whose shape also depends on *B* − *A*, which mean curvature does not carry. The sum term is kept because it is exact for the cases the model is most often applied to and needs one number per body rather than a curvature tensor and a relative orientation.

**The shared polygon has the same sensitivity, and the patch cap absorbs it.** The exact collision shape of a curved convex body is its convex hull, which is faceted, so one contact can report a real curvature and a facet polygon at the same time. That is what exactness produces rather than what a proxy produces, and no better collider removes it. It needs no guard of its own, because [the patch radius](#one-contact-stiffness-capped-by-the-shared-polygon) takes the smaller of the two: under light load the growing Hertz patch is far narrower than a facet, so the curvature decides and the facet is never reached, and under a load heavy enough for the patch to span a facet the facet decides, which is what a body bearing flat over one does. Reading the polygon on its own, with no curvature to cap it, would give a hulled sphere the contact time of a small flat punch.

## One modal model per rigid body, but a surface per collider

**Decision.** A contact resolves both the modal model and the acoustic surface from the collider node it touched, or its nearest ancestor carrying one. At most one node of a rigid body's hierarchy may carry a model. Any number may carry a surface.

**Why the walk starts at the collider.** A machine on rubber feet with a steel shell is one body, and a contact on a foot must find rubber where one on the shell finds steel. Finish, elastic constants, and curvature all belong to the geometry touched.

**Why only one model.** Modes are the eigen-decomposition of one connected structure and span every sub-shape of a compound. A hammer's head and handle are rigidly joined, so striking the head drives modes reaching through the handle. A collider hierarchy decomposes collision geometry, not the vibrating object. Models per collider would give one body several uncoupled resonators and make its spectrum depend on which part was struck.

**Why scoped to rigid bodies.** Nesting alone says nothing about elasticity: a shelf with cups parented under it wants a model per cup. Only a rigid body asserts that its parts move as one.

## One body, one mass: shapes renormalize to the authoritative mass at fixed frequencies

**Decision.** Every mass-consuming path (contact dynamics, Hertz contact time, acceleration noise, and the modal shapes' normalization) reads the body's one authoritative mass. When a model's shapes were derived against a different solve mass, they scale by √(*M*ₛ/*M*) and the frequencies stay.

**Why one mass.** A body's authored mass and the solid mass its geometry implies at the material's density can disagree by orders of magnitude, as when a physics demo authors a large box at a nominal 1 kg. The dynamics, the recoil, the contact time, and the shapes' normalization each consume a mass, and relative loudness across those paths is meaningful only when they all read the same one.

**Why frequencies stay.** Scaling shapes at fixed frequencies preserves the material's frequency character while making the struck response carry the energy an impulse delivers to a body of mass *M*. It is exactly the solve of a body whose density and stiffness both scale by *M*/*M*ₛ, a lighter material at the same specific stiffness *E*/ρ. Specific stiffness varies far less across real materials than density does, so this is the least-wrong homogeneous reading of "this geometry, this material character, that mass". It is also the first-order account of a hollow body of the material: a shell rings in a similar band to the solid, responds harder per impulse, and keeps the solid's surface hardness, which is why the contact time keeps the material's true *E*.

**Why not rescale density at fixed stiffness.** That reading shifts every frequency by √(*M*ₛ/*M*). The 1 kg box's spectrum would move up 27 fold, mostly past hearing, and the implied material (ceramic stiffness at balloon density) exists nowhere. It also silences exactly the bodies whose response should grow.

**Why not solve the true shell.** A body lighter than its solid is physically a shell or composite, and a shell eigensolve is the exact answer. The asset does not say where the material is, wall thickness from mass alone is often absurd, and the renormalized solid is a bounded first-order stand-in. An author who wants shell acoustics can solve them and ship the resulting model, which the spec's data layout already carries.

**Energy consistency.** With shapes normalized to *M*, an impulse *j* deposits modal energy on the scale of *j*²/2*M*, the same budget the rigid recoil draws from.

## Surfaces are absolute, and separate from materials

**Decision.** Acoustic surfaces are their own array, not fields on the acoustic material, and their lengths are absolute physical quantities exempt from node scaling.

**Why separate.** Bulk and finish vary independently: polished and sandblasted steel share every material constant and sound completely different when scraped. Merging them would duplicate five material constants per finish and, worse, create a precedence question when a node's surface and its model's material disagree. A surface referencing a material has no such conflict, and gives a silent floor a way to carry the elastic constants that contact stiffness needs.

**Why exempt from scaling.** Modal data describes an object at one physical size and rescales with it, but a finish does not: a scaled-up polished sphere is still polished. Contact position and sweep speed are geometric and do transform, and they keep their magnitudes when a contact state is expressed in node-local space, since a node-scaled velocity would read an absolute finish at the wrong rate. The exemption covers the microscale parameters only. Mesoscale relief is bound to texture coordinates rather than to an absolute length, so it scales with the node like the geometry it is painted on, and a scaled-up tiled floor has larger tiles to both eye and ear.

## Two surface scales, with the mesoscale bound to mesh UVs

**Decision.** A surface carries statistical microscale parameters plus an optional `normalTexture` for mesoscale relief, interpreted as glTF core's `normalTextureInfo` and sampled along the contact path. Absent, it falls back to the contacted primitive's material `normalTexture`, so the surface property overrides the correspondence rather than establishing it.

**Why a fallback rather than a recommendation.** The map wanted is almost always the one the material already references, and a recommendation makes the common case the effortful one while leaving the two references free to drift: texture references are positional indices, so swapping a material's map or reordering the `textures` array silently desynchronizes them with nothing to detect it. A fallback also carries the existing asset corpus, where normal maps are everywhere and this extension is nowhere: 55 of the 148 Khronos sample models declare a material `normalTexture`, and under a fallback every one of them sounds like what it looks like the moment it gains an acoustic surface. The whole `normalTextureInfo` triple is inherited, including `scale`, because `scale` multiplies the surface gradient and is interchangeable with the map's own contents. A map authored at half strength with `scale` 2 is identical to that map at full strength with `scale` 1, so there is no line between the physical map and an artistic multiplier, and inheriting only part of the triple would make the audible relief a different amplitude from the visible one.

**Why per primitive.** A glTF mesh is split into primitives to give its parts different materials, so a multi-primitive mesh is an object whose surface genuinely differs from region to region, and one arbitrary primitive is the wrong answer everywhere else. Texture coordinate sets settle it independently of taste: they are declared per primitive, so a `texCoord` index inherited from one primitive's material may not even exist on the geometry being touched. Resolving both against the primitive containing the contact keeps the pair coherent with no tie-breaking rule.

**Why two scales.** Ren et al. 2010 decompose contact surfaces into macro geometry, mesoscale bumpiness carried by normal maps, and microscale roughness, and state the gap directly: fractal noise alone "does not render any information for the bumpiness or heterogeneous variation of the contacting geometry at the meso level," which is "clearly visible to the users but transparent to the rigid-body simulator." Tiling, grout, corrugation, knurling, and grain live in exactly that band. The collision mesh is too coarse for them and a statistical finish is orders of magnitude too fine.

**Why it matters most for rolling.** The contact patch filter selects between the scales. A 1 cm steel ball on steel under 1 N has a Hertz radius near 40 µm, so a micron-scale finish sits at or below the patch and is strongly attenuated, while millimetre-scale relief passes intact. Rolling hears the tiles and not the polish. Ren's own mechanism is a normal-direction impulse, which is precisely the channel that survives at zero slip, so the mesoscale layer strengthens rolling more than scraping.

**Why UV binding is acceptable here.** The spec argues against binding *modal* data to meshes because mode shapes come from a tet solve with no correspondence to render topology. Surface relief is the opposite: it is a property of the rendered surface, it already exists in the asset as a UV-mapped image, and UV-mapped textures are how glTF expresses spatially varying surface properties everywhere else. Referencing the material's own normal map is the point rather than a compromise, because it makes the sound track what is visibly being crossed.

**Why this was available and unused.** Ren's meso contribution did not propagate. Later work in this line cites that paper for its source-filter architecture and for fractal-noise friction alongside van den Doel, never for the mesoscale layer. The idea needs a graphics pipeline to read a normal map from, which perceptual research using measured depth profiles does not have. glTF is the carrier it was missing.

**Consequence for scope.** Spatial variation and directionality are now expressible at the mesoscale, so only the microscale parameters remain isotropic and uniform per node.

## Waviness is a third surface scale, not a wider finish

**Decision.** A surface carries `waviness` and `wavinessLength` alongside the microscale statistics, entering only through their ratio, the mesoscale gradient. It reduces the region two nominally flat faces bear over, and the finish is resolved inside what it leaves. The spec requires it to be the longer and gentler of the two scales.

**Why a separate scale rather than a wider correlation length.** The two do different work. σ and *ℓ* describe the asperities that carry the load and set the rate a contact crosses them at. Waviness describes whether the faces are close enough to touch at all: two nominally flat faces are never flat, and the departure the process leaves holds most of their shared polygon apart. Folding it into the finish would make one gradient answer both questions, and they have different answers: a cast face is rough at 1e-4 m over 1e-3 m while its mould form is 2e-4 m over 2e-2 m, gentler by an order.

**Why it matters audibly.** Without it, a large flat face bears over its whole polygon, which puts so many asperities under load that their fluctuations average away and the contact renders silent. The area a load actually bears over is smaller than the polygon by orders of magnitude, and the count of load-bearing asperities follows it. Applying the area law once at each scale, which is Persson's magnification in two discrete steps, recovers a bearing population small enough to fluctuate.

**Why the ordering is required rather than recommended.** Applying the same area reduction twice at one scale is not a degraded result but a meaningless one: the second application is describing the finish a second time under another name. Making it a MUST lets an implementation trust the two scales are separated instead of testing for it.

**Why absent means flat.** A surface with no waviness bears across the whole shared polygon, which is the idealized geometry contact-mechanics simulations use and the limit the two-step reduction collapses to. That keeps the field optional without leaving its absence undefined.

**Revisit when** a surface wants anisotropic waviness, or waviness that varies across a face. Both are the same generalization the microscale parameters would need, and the mesoscale `normalTexture` already covers spatial variation one scale below.

## The element container carries the mesoscale as a band of its own field, a relief map included

**Decision.** Under the element container the mesoscale enters as its own band beside the finish, drawn once over the whole ribbon and read as one lift per element, so the elements standing on its crests carry the load and the face-wide envelope bridges its valleys. A side carrying a relief map takes that band from the map's own heights, at the band those heights hold, and the waviness parameters stand in only where no map does. A relief map also keeps driving the contact's own datum through its track, which is a separate job the field cannot do for a wide contact.

**Why the field rather than an area reduction.** The statistical route reduces the region two faces bear over by the mesoscale gradient and resolves the finish inside what is left, which is the right answer when the finish is all that is resolved. The container resolves the mesoscale instead, and the two are alternatives rather than a sequence: applying the area law to a scale the field already carries is the same double count the waviness ordering forbids one scale up. Resolved, the mesoscale does what a closure can only approximate, since which asperities bear becomes a matter of where the crests are rather than how much area survives, and it varies across one footprint.

**Why a relief map belongs in that band rather than beside it.** A relief map is the realization of the same departure from flat the waviness parameters describe statistically, so it is the same scale and belongs in the same slot. Carrying it only as a scalar height signal left it unable to decide how much of a face bears or to vary across a footprint, which are the two things a mesoscale is for, and it would have made that permanent for every relief-mapped body once the statistical path was gone. Its band is measured off its own heights rather than authored, since a map ships no parameters beside it.

**What the map supplies, and what it does not.** The relief track is sampled along a fixed path across the map rather than along the contact's own, so its heights are a representative trace of the map and not a positioned one. Its band is therefore what it can honestly supply, and the across-ribbon direction is synthesized from that trace's own spectrum exactly as a measured profile's is. Sampling the map in the contact's own frame is the standing improvement, and it would make the lift positional as well as varied.

**Why the track still drives the datum, and the measurement that settles it.** The field is periodic over the length the elements tile, and a contact whose footprint spans that length reaches every element, so its support envelope is a maximum over the whole field and never moves as the contact sweeps. Such a contact renders no datum motion from the field at all, whatever the field carries: measured on a face 12 cm across a field 10 cm long, the mesoscale channel falls **50 dB** when the track's datum motion is removed in favour of the field's. The track is therefore the datum's channel and the field is bearing's, and the two overlap only for a contact whose footprint is small against the field, where the field's own relief would move the datum as well. No authored scene reaches that regime, and with the track's datum kept the three relief scenes move **+0.5, +0.2 and −0.7 dB**, which is the bearing change alone.

**Revisit when** a contact's footprint is small against the field's sweep, where the mesoscale would move the datum twice. The band boundary is available to split them, since the datum channel already has its own DC length, but a scale split chosen for that reason is arbitrary until a scene needs one. The end state is the lift applied when the springs are read rather than when they are built, which makes the relief positional and leaves the datum channel nothing to carry.

## Framed by mechanism, not by perceptual category

**Decision.** Sustained contact is specified as *roughness excitation*, one mechanism, with scraping and rolling as names for regions of a continuous parameter space. Impact excitation and friction-induced vibration are the other two mechanisms, one defined elsewhere in the spec and one excluded.

**Why.** The computer-graphics-audio literature frames this phenomenologically, as scraping and rolling, and taking those categories structurally causes real failures. Ren et al. classify contacts by whether tangential velocity is nonzero, which routes pure rolling away from their surface model entirely. The same habit collapses per-body sweep into one speed, which silences a box sliding on a floor: the box's own sweep is zero because the same material region stays in contact, and only the floor's is nonzero.

Two engineering fields converge on a mechanism framing, independently of each other and of the graphics literature. Rolling-noise prediction for railways and tyres has treated surface roughness as the excitation since Remington 1987. Friction acoustics calls the same phenomenon *roughness noise*, arising from asperity impacts in light contact, and separates it from the instability-driven noise of strongly loaded contact (Akay 2002). That second split is our scope boundary, arrived at by a field that had no interest in our problem.

**The rolling-noise literature also supplies the standard name for the patch attenuation.** What this spec derives from the Hertz contact radius is the *contact filter*, where roughness with wavelengths shorter than the contact patch is attenuated. That literature independently confirms the cutoff falls with speed, and that a softer contact filters more strongly, which is what deriving the patch radius from the effective modulus predicts.

**And it independently supports treating the excitation as a vector.** Sliding surfaces are observed to develop contact forces with components in both tangential and normal directions, each driving its own response, and the partitioning between them varies with normal load (Akay 2002). A scalar force along the contact normal cannot express that.

## Rolling emerges rather than being modelled

**Decision.** No rolling-specific force term. Rolling is the same contact with zero slip, and the patch filter runs on every contact rather than being switched on for rolling.

**Why.** Attribution matters here because the pieces come from different places. van den Doel et al. 2001 identify the mechanism, attributing rolling's low-frequency character to collisions just in front of a very small contact area, and relate collision velocity to the contact region, but leave the cutoff as a free parameter to tune. Deriving it from the Hertz patch radius closes that parameter using constants the contact stiffness already needs, and drops the assumption that the rolling body is a sphere.

Agarwal et al. 2021 supply the force algebra and a rolling term from the offset between a ball's center of mass and its geometric center. That term is omitted: it presumes a sphere. They also note that the balance between scraping and rolling contributions "likely depends on how much slip is present" and then leave it to hand-set free parameters. Their motions are prescribed analytically rather than simulated, so slip is not a quantity their model produces. A rigid-body solver computes slip and sweep exactly, which turns their free parameter into a derived quantity. That is the strongest argument for reporting two speeds rather than one.

**A one-speed classifier fails here.** Ren et al. classify a contact as lasting when relative tangential velocity is nonzero and transient otherwise, then apply their surface representation only to lasting contacts. Pure rolling has zero tangential velocity, so their own criterion routes it away from the surface model and into an impulse sequence generated at tessellated-geometry contacts, which makes rolling sound vary with mesh density. Slip alone cannot separate resting from rolling.

**Deliberately not duplicated.** Restitution and friction stay in `KHR_physics_rigid_bodies`. Unlike mass, where acoustic and gameplay values legitimately differ, restitution has no acoustic reading distinct from its mechanical one, so a second copy would only create conflicts.

**Naming hazard documented.** A render material's `roughness` is dimensionless optical microfacet statistics tuned by eye. This one is a physical length measured with a profilometer, orders of magnitude coarser. Neither derives from the other, and the spec says so where an author will look.

## Traversal is indexed by distance along the path

**Decision.** A surface's track is read at the distance the contact has travelled along that surface. A contact that retraces its path reads fresh surface rather than the heights it came from.

**Why not signed displacement.** Signed displacement is right about the physics and unreachable in the data model. A track is a height field, so retracing should re-read it, and the excursion of a rocking body should sound like a rattle rather than a hiss. But indexing by a *signed* quantity needs a direction to be signed against, and a one-dimensional track has none. Every candidate fails on a path a rigid-body scene can produce: the instantaneous sweep direction is never negative relative to itself, a fixed axis is silent for motion across it, a smoothed reference introduces a time constant that has to sit between the reversal and curvature timescales, and a separable pair of readers makes the excitation 3 dB louder diagonally than axially. A requirement written in terms of "traversal direction" names something a one-dimensional track cannot define.

**What distance gets right, which is more than it looks.** It is the only candidate with no direction dependence at all, in amplitude or in correlation, and it needs no parameter. The rate of asperity encounters scales with speed irrespective of sign, so it recovers the right bandwidth and the right level. It gets the fine structure wrong on a retracing contact and nothing else wrong anywhere. A track models a path, not a surface, and distance is the faithful reading of that model.

**What removing the limitation takes, and it differs by scale.** The two layers are not in the same position.

The **microscale finish** has no two-dimensional data to read. A stored field works over an extent matched to the contact's excursion, which is what Agarwal et al. 2021 do with a confocal depth map for a hand-held scraper, and that paper is two-dimensional where this spec's `profile` is not. Covering unbounded sliding instead costs (*T* *f*<sub>Nyquist</sub>)² samples for *T* seconds of non-repeating output, around 6 × 10⁸ for one second at 48 kHz, independent of speed because extent and resolution both scale with it and cancel. That cancellation is also why the one-dimensional track needs only *T* *f*<sub>Nyquist</sub> samples, the figure this spec already quotes. Procedural evaluation stores nothing and synthesizes only the octaves the current speed and load resolve, making synthesis and the patch filter one operation, at a per-sample cost against a prefix-sum lookup that is a measurement rather than an argument.

The **mesoscale relief** is already a field: `normalTexture` is a two-dimensional texture and the contact state already carries **p** to sample it at. Its costs are bounded and known rather than open. Measured over the Khronos sample assets, a single-channel height field with a full mip pyramid is 11 MB for a 2048² map and under 1 MB for more than half of them. A screened Poisson solve at derivation time gives the height from the normal map path-independently, which is the two-dimensional form of the leak this spec already prescribes and removes the drift that motivates it, leaving the tangential term to read the map's slope directly with no integration at all. Reading it at **p** rather than along a path is the better model, and the reason this spec does not yet require it is that obtaining a texture coordinate at the contact needs the contacting geometry to be the textured mesh, which a simplified collider is not.

## Tangential force split by mechanism

**Decision.** The tangential excitation is two terms, not one. A geometric term, the contact load projected onto the locally tilted surface, acting along each surface's own sweep direction. A frictional term, Coulomb traction riding on the force fluctuation, acting along the slip direction. Neither carries a free parameter.

**Why.** Riding over a tilted asperity converts normal load into tangential force whether or not anything slides, and it acts along the direction of travel over that surface. Coulomb traction acts along the slip and is bounded by *μN*. They have different directions, so one term cannot carry both.

**Their v is the contact's velocity across the surface, not the slip.** Agarwal et al. 2021 write *f*<sub>h</sub> = β₁|**v** · ∇*S*|^β₂. Reading that **v** as the slip makes the term vanish for pure rolling, which their own rolling equation contradicts by keeping it. That one substitution is the difference between rolling sounding and rolling falling silent.

**And their term is not a separate mechanism, it is this one's velocity-dependent half.** Because *f*<sub>n</sub> carries the Hunt and Crossley damping factor, *f*<sub>n</sub> ∂*h* expands to *k*δ^(3/2) ∂*h* + *k*δ^(3/2) *c*<sub>d</sub> δ̇ ∂*h*, whose second part is proportional to the rate the height under the contact changes, which is what their term measures. Matching them identifies β₁ with *f*<sub>n</sub> *c*<sub>d</sub> ∂*h*. That product carries exactly the N·s·m⁻¹ their constant must have for the result to be a force, which the source leaves unstated, and it names what a fitted value absorbs: the load, the material's dissipation constant, and the surface slope. So projecting the whole contact force is not a rival model, it is the parameter-free form of theirs, and it is why the constant needed refitting per surface. The same move closed van den Doel's rolling cutoff and Agarwal's slip balance.

**Consequence for the contact state.** The force model acts along directions rather than magnitudes, so the state carries slip and sweep as velocities, and the speeds the requirements are written in terms of are their magnitudes.

**The geometric term reverses per surface, which reads as a sign convention and is a physical result.** Take the one arrangement where the interface is unambiguous: a smooth box sliding on a bumpy floor. The box has no sweep of its own, since the same material region stays in contact, so only the floor's surface is traversed, and the contact normal is simply the floor's. The box climbing a rising asperity is retarded and the floor is driven forward. Swapping which body carries the asperity, a bumpy ball rolling on a smooth floor, is the same mechanics with the roles exchanged and so demands the opposite sign for the same slope. A body is therefore driven along **û**ᵢ by its own surface's slope and against it by the other body's, and the two bodies receive equal and opposite excitations as Newton's third law requires. Summing the two surfaces with one sign, which is the form that first suggests itself from *h* = *h*₁ + *h*₂, satisfies neither arrangement.

**Not the average of the two boundaries.** Treating the interface as the mean of the two surfaces gives a slope of (∂*h*₁ − ∂*h*₂)/2, which has the right antisymmetry and the wrong magnitude: a smooth body rides exactly on a rough one's profile, with no halving. Superposing the two single-rough-surface cases is the right construction, and is the same linearization the rest of the roughness treatment rests on.

**The audible reach is narrow, and the correctness is not.** Within one body the two surfaces' slopes are independent, so a difference and a sum have the same root-mean-square and an uncorrelated pair sounds the same either way. It is when both bodies of a contact sound that a common sign adds two excitations that should cancel.

**Which forces the frames to be settled.** Speeds are frame-independent, so a wholly node-local state can write the slip as a difference of the two sweeps and stay consistent. As vectors that subtraction spans two different nodes' frames and means nothing, so the kinematic relations are stated in a common frame and each body's state is transformed into its own. The transform is the impulsive path's: position by the full inverse, directions and velocities by the inverse rotation alone. Velocities keep their magnitude for a physical reason rather than a tidy one, since `sampleSpacing` is absolute and a scaled node would otherwise read its finish at the wrong rate.

## Vibrational coupling is MAY

**Decision.** Coupling the contact force to the body's own modal displacement is optional.

**Why.** It is the mechanism behind micro-collisions, chattering, and contact-dependent damping (Zheng and James 2011), and an open-loop force cannot produce them. But it is a real cost: the force at each sample depends on the summed modal state at the previous one, which serializes what is otherwise a per-mode-parallel update, and its absence degrades gracefully to the open-loop model that has been standard since FoleyAutomatic. MAY is the honest level.

**It does not break linear superposition.** The nonlinearity is in the force *generator*, upstream of the mode bank, which stays linear. The existing MUST on superposing excitations is about the bank and survives unchanged. Worth stating explicitly because "nonlinear vibration" is an exclusion elsewhere in the spec.

**Bilateral coupling is left open.** When both bodies sound, a full treatment shares one separation between two mode banks. The spec permits coupling each body only to itself, which is what a per-object renderer will do.

## Out of scope, and why each exclusion is safe

Everything excluded *composes with* the modal core rather than replacing it:

- **Radiation transfer** — exactly separable: per-mode scalar amplitude field multiplying oscillator output (uncontested from PAT through KleinPAT). WaveBlender proves the same modal payload drives a full wave solve with no extra data. Future layer: per-mode FFAT-style cube maps (~2 KB/mode as 8-bit textures) map naturally onto glTF's image machinery.
- **Propagation/spatialization/listeners** — `KHR_audio_graph`/`KHR_audio_emitter`/platform territory; this extension only produces a monophonic source at the node.
- **Contact-event plumbing** — matches KHR_physics_rigid_bodies' own scoping.
- **Friction-induced vibration** (squeal, bowing, brake noise) — velocity-weakening friction driving a limit cycle, which needs its own bristle state and is a different mechanism from riding over asperities. Composes as an additional force term on the same contact.
- **Anisotropic and spatially varying microscale finish, and surface wear** — the statistical parameters are isotropic and uniform per node, with variation and directionality expressible one scale up through `normalTexture`. Wear would make a surface dynamic, which conflicts with static document data.
- **Thin-shell mode coupling, fracture, sample hybrids** — known limitations of linear modal synthesis, documented as validity bounds (Authoring Notes) rather than half-specified features.
- **Analysis itself** — authoring-time; appendix documents the standard pipeline informatively.

**Declarative data with fixed semantics.** The spec ships baked results and defines what a renderer does with them. Every surviving system in the survey works this way, and adoption tracks it: MPEG-4 Structured Audio, which standardized a programmable synthesis language instead, saw none.

## Optionality and fallback summary

- Extension is purely additive: `extensionsUsed` only; assets SHOULD NOT list it in `extensionsRequired`.
- Nothing about sustained contact is required of a modal renderer. An implementation that renders only impacts is conformant.
- Required per model: `frequencies`, `decayRates`, `positions`, `shapes`. Nothing is required on the node object, and all three root arrays are optional.
- Optional: `indices` (fallback: nearest-point), `material` (fallback: no rescaling/re-derivation/contact-duration estimation), `massProperties` (fallback: physics extension or watertight geometry + ρ), `gain` (default 1.0), `name`.
- Optional per surface: everything. `roughness`, `correlationLength`, and `spectralSlope` default to a machined finish; `profile` falls back to a synthesized track; `normalTexture` falls back to the contacted primitive's material `normalTexture`, and only a surface with neither is smooth at the mesoscale; `material` falls back to the node's model's material.
- Renderer latitude: mode-subset rendering MAY, vibrational coupling MAY, barycentric interpolation SHOULD, uniform-scale adjustment SHOULD, duration shaping SHOULD, acceleration noise SHOULD, sustained contact SHOULD, contact-force behavioral properties SHOULD; aliasing prohibition, linear superposition, per-instance state, non-negative contact force, silence at rest, the microscale finish exempt from node scale, and mesoscale relief sized by it MUST.
- Every MUST forbids an audible failure mode (aliased partials, crosstalk between excitations, a shared oscillator, a contact that pulls, a resting body that hums, a finish that changes when an object is resized, a relief that does not) rather than mandating an algorithm.

## Deferred / future work

- Example + conformance test assets (deliberately after spec stabilization).
- Radiation-transfer layer (FFAT cube maps or multipole coefficients) as a sibling extension.
- Sample-based response models (recorded per-strike-point impulse responses) as a peer to modal, once audio resources exist to reference. This is the trigger for splitting `KHR_audio_materials` out and nesting node roles.
- Friction-induced vibration (stick-slip squeal) as an additional force term on the same contact.
- Mode-shape compression (Langlois-style) if asset sizes demand it.
- `KHR_interactivity` excitation node (contact events carrying impulse + position, and sustained contact state) once that ecosystem settles.
- MeshEditor conformance: retain eigenvector 3-vectors in `mesh2modes`, import/export of the extension, direction-aware excitation, sustained-contact rendering.
