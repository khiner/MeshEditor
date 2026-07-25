# KHR_audio_rigid_bodies — Design Decisions and Rationale

Companion reference for the spec in `extensions/2.0/Khronos/KHR_audio_rigid_bodies/`. Not part of the specification. Records what the spec decides, why, and what would prompt revisiting.

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

## Sustained contact is one excitation regime, not a second model

**Decision.** Sliding, scraping, and rolling are rendered as a continuous force driving the same modes, mode shapes, and mass properties an impact drives. Rolling and sliding are not separate models.

**Why.** The physics distinguishes them by measurable speeds, not by a category. Each body has its own sweep speed, the rate the contact travels over that body's surface, and slip is the difference between the two. Pure rolling has both sweeps equal so slip vanishes; a box sliding on a fixed floor has zero sweep on the box and sweep equal to slip on the floor; partial slip lies between, and a body that rolls then skids moves continuously across that range.

**Sweep is per body, which is easy to get wrong.** Collapsing the two into a single sweep speed silences the canonical scraping case: the sliding box's own sweep is zero, since the same material region stays in contact, so a single-rate traversal never advances and the patch cutoff collapses to zero. Each surface must be traversed at its own rate and the contributions summed. A body sounds because the *other* surface streams past its stationary contact patch. Any implementation that switches between a "rolling model" and a "sliding model" introduces an audible transition at a place where the physics has none, which is why the spec says SHOULD NOT rather than leaving it open. Both speeds fall out of a rigid-body solver at no extra cost, so this needs no estimation heuristics.

**Implication for the excitation definition.** The impulse form is the limit of the sustained form as contact duration goes to zero, which the spec states. A renderer driving modes with a sampled force pulse (SHOULD-level, for contact-duration shaping) renders sustained contact by substituting a different force signal, with no other change.

## Contact force: constrain behavior, not formula

**Decision.** The spec states properties any conformant contact force must have (non-negative, silent at rest, traversal indexed by distance, √v·N amplitude, rolling low-passed by patch radius, force limited by load) and puts one satisfying model in a non-normative appendix.

**Why.** This matches how [Synthesis](#synthesis-normative-relative-levels-free-implementation) is already scoped: pin the perceptible behavior, leave the architecture free. Mandating Hunt and Crossley would exclude equally valid formulations, and mandating nothing would let a renderer emit fixed-rate noise that neither tracks speed nor stops, which is qualitatively wrong rather than merely different.

**Two requirements are MUST**, on the same reasoning as the Nyquist rule: each forbids exactly one audible failure and permits every real architecture. Non-negative force forbids a contact that pulls, and preserves the separation nonlinearity that produces micro-collisions and chatter. Silence at rest forbids a settled body humming forever, which is the single worst failure mode for this feature and the one a naive implementation falls into by leaving the equilibrium Hertz force in the excitation.

**Parametric surfaces are statistical, stored profiles are reproducible.** A surface given only by roughness, correlation length, and spectral slope specifies an ensemble, so two renderers agree on character but not sample-for-sample. The optional `profile` accessor is the escape hatch, and it also matches the format's philosophy of shipping baked results rather than standardizing a generator (the MPEG-4 Structured Audio anti-pattern noted below). Measured tracks are small: a few thousand samples at a few microns covers enough surface to loop inaudibly.

## Surfaces are absolute, and separate from materials

**Decision.** Acoustic surfaces are their own array, not fields on the acoustic material, and their lengths are absolute physical quantities exempt from node scaling.

**Why separate.** Bulk and finish vary independently: polished and sandblasted steel share every material constant and sound completely different when scraped. Merging them would duplicate five material constants per finish and, worse, create a precedence question when a node's surface and its model's material disagree. A surface referencing a material has no such conflict, and gives a silent floor a way to carry the elastic constants that contact stiffness needs.

**Why exempt from scaling.** Modal data describes an object at one physical size and rescales with it, but a finish does not: a scaled-up polished sphere is still polished. Contact position and sweep speed are geometric and do transform.

## Two surface scales, with the mesoscale bound to mesh UVs

**Decision.** A surface carries statistical microscale parameters plus an optional `normalTexture` for mesoscale relief, interpreted as glTF core's `normalTextureInfo` and sampled along the contact path.

**Why two scales.** Ren et al. 2010 decompose contact surfaces into macro geometry, mesoscale bumpiness carried by normal maps, and microscale roughness, and state the gap directly: fractal noise alone "does not render any information for the bumpiness or heterogeneous variation of the contacting geometry at the meso level," which is "clearly visible to the users but transparent to the rigid-body simulator." Tiling, grout, corrugation, knurling, and grain live in exactly that band. The collision mesh is too coarse for them and a statistical finish is orders of magnitude too fine.

**Why it matters most for rolling.** The contact patch filter selects between the scales. A 1 cm steel ball on steel under 1 N has a Hertz radius near 40 µm, so a micron-scale finish sits at or below the patch and is strongly attenuated, while millimetre-scale relief passes intact. Rolling hears the tiles and not the polish. Ren's own mechanism is a normal-direction impulse, which is precisely the channel that survives at zero slip, so the mesoscale layer strengthens rolling more than scraping.

**Why UV binding is acceptable here.** The spec argues against binding *modal* data to meshes because mode shapes come from a tet solve with no correspondence to render topology. Surface relief is the opposite: it is a property of the rendered surface, it already exists in the asset as a UV-mapped image, and UV-mapped textures are how glTF expresses spatially varying surface properties everywhere else. Referencing the material's own normal map is the point rather than a compromise, because it makes the sound track what is visibly being crossed.

**Why this was available and unused.** Ren's meso contribution did not propagate. Later work in this line cites that paper for its source-filter architecture and for fractal-noise friction alongside van den Doel, never for the mesoscale layer. The idea needs a graphics pipeline to read a normal map from, which perceptual research using measured depth profiles does not have. glTF is the carrier it was missing.

**Consequence for scope.** Spatial variation and directionality are now expressible at the mesoscale, so only the microscale parameters remain isotropic and uniform per node.

## Framed by mechanism, not by perceptual category

**Decision.** Sustained contact is specified as *roughness excitation*, one mechanism, with scraping and rolling as names for regions of a continuous parameter space. Impact excitation and friction-induced vibration are the other two mechanisms, one defined elsewhere in the spec and one excluded.

**Why.** The computer-graphics-audio literature frames this phenomenologically, as scraping and rolling, and taking those categories structurally causes real failures. Ren et al. classify contacts by whether tangential velocity is nonzero, which routes pure rolling away from their surface model entirely. The same habit produced a defect in an earlier draft of this spec, where collapsing per-body sweep into a single speed silenced a box sliding on a floor.

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
- Optional per surface: everything. `roughness`, `correlationLength`, and `spectralSlope` default to a machined finish; `profile` falls back to a synthesized track; `normalTexture` absent means smooth at the mesoscale; `material` falls back to the node's model's material.
- Renderer latitude: mode-subset rendering MAY, vibrational coupling MAY, barycentric interpolation SHOULD, uniform-scale adjustment SHOULD, duration shaping SHOULD, acceleration noise SHOULD, sustained contact SHOULD, contact-force behavioral properties SHOULD; aliasing prohibition, linear superposition, per-instance state, non-negative contact force, silence at rest, and surfaces exempt from node scale MUST.
- Every MUST forbids an audible failure mode (aliased partials, crosstalk between excitations, a shared oscillator, a contact that pulls, a resting body that hums, a finish that changes when an object is resized) rather than mandating an algorithm.

## Deferred / future work

- Example + conformance test assets (deliberately after spec stabilization).
- Radiation-transfer layer (FFAT cube maps or multipole coefficients) as a sibling extension.
- Sample-based response models (recorded per-strike-point impulse responses) as a peer to modal, once audio resources exist to reference. This is the trigger for splitting `KHR_audio_materials` out and nesting node roles.
- Friction-induced vibration (stick-slip squeal) as an additional force term on the same contact.
- Mode-shape compression (Langlois-style) if asset sizes demand it.
- `KHR_interactivity` excitation node (contact events carrying impulse + position, and sustained contact state) once that ecosystem settles.
- MeshEditor conformance: retain eigenvector 3-vectors in `mesh2modes`, import/export of the extension, direction-aware excitation, sustained-contact rendering.
