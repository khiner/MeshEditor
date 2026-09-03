# KHR_audio_rigid_bodies — Design Decisions and Rationale

Companion rationale for the specification in `extensions/2.0/Khronos/KHR_audio_rigid_bodies/`. This document is non-normative.

The sections through *Sustained contact excitation* cover modal models, acoustic materials, impact excitation, synthesis, acceleration noise, and radiation. Later sections cover the evolving sustained-contact design.

## Name and positioning: `KHR_audio_rigid_bodies`

**Decision.** Use a KHR-prefixed name, Draft status, and the `extensions/2.0/Khronos/` layout, following the `KHR_physics_rigid_bodies` precedent. The repository name `glTF_PhysicalAudio` identifies the domain.

**Scope.** Khronos names extensions `<PREFIX>_<scope>_<feature>`. The `KHR_audio_graph` and `KHR_audio_emitter` proposals establish the `audio` scope for sources, emitters, listeners, and spatialization. This extension generates signals for those systems. A survey of Khronos, OMI, MPEG, and middleware found no conflicting modal or procedural-audio glTF proposal.

**Feature name.** `rigid_bodies` identifies the subject and its main assumption. The extension also includes acceleration noise, `massProperties`, acoustic surfaces, and contact forces, so a synthesis-specific name would be incomplete. The name associates the extension with the bodies and contact events described by `KHR_physics_rigid_bodies`.

**Discoverability.** The Overview includes “modal sound” because the extension name does not.

## Extension scope

**Decision.** Keep models, acoustic materials, acoustic surfaces, synthesis, acceleration noise, and sustained contact in `KHR_audio_rigid_bodies`.

`physicsMaterials`, `physicsJoints`, and `collisionFilters` are root-level arrays inside `KHR_physics_rigid_bodies`. `KHR_implicit_shapes` is separate because shape data has prospective consumers outside rigid-body physics. A separate extension requires a demonstrated independent consumer.

Only contact sound consumes acoustic surfaces. Acoustic materials also provide `alpha` and `beta` for body resonance.

Contact state, the composite-surface rule, and the force model add no independent JSON payload, so they remain a specification section.

**Revisit when.** Split out `KHR_audio_materials` if a second consumer of surfaces or materials appears, such as sample-based response models.

## Optional node roles

**Decision.** `modalModel`, `acousticSurface`, and `gain` are all optional on the node extension object. A node opts into whichever roles apply.

Sustained contact is pairwise, and a silent floor or table may contribute only its finish. `node.KHR_physics_rigid_bodies` uses the same optional-role structure for `motion`, `collider`, `trigger`, and `joint`.

**Array names.** Prefix `modalModels`, `acousticMaterials`, and `acousticSurfaces`, following `physicsMaterials` and `collisionFilters`. Bare `models` is ambiguous, and bare `materials` conflicts conceptually with core glTF materials.

**Ratification.** The expected sequence is vendor prefix, EXT after a second independent implementation, and KHR after a Khronos vote. The glTF_Physics precedent permits a KHR name before ratification. Draft review requires a JSON schema, test assets, and a third-party implementation; test assets follow specification stabilization.

## Combined data and behavior

**Decision.** A single extension carries both the data (models, materials) and the rendering semantics.

Modal model data currently has one consumer: a modal audio renderer. Revisit this decision if another extension, such as haptics, consumes the same modes.

## Root arrays and node references

**Decision.** `modalModels[]`, `acousticMaterials[]`, and `acousticSurfaces[]` live in a `KHR_audio_rigid_bodies` object on the glTF root (items inherit `glTFChildOfRootProperty`, so they get `name`); nodes instance a model and a surface by index with an optional per-instance `gain`.

This follows the registry pattern used by `KHR_lights_punctual` and physics materials. Nodes share resource data but maintain independent oscillator state. Node-level `gain` controls each instance, following JASS and Phya global scale factors.

## Vector mode shapes

**Decision.** The canonical spatial payload is per-mode, per-sample-point *displacement vectors* φ (mass-normalized eigenvectors). Excitation is the projection aₙ = φₙ(p) · j.

Scalar per-position gains make tangential and normal impulses at one point produce identical excitation. Vectors preserve direction-dependent timbre and provide the normal-projected surface velocity required by radiation models. O'Brien 2002, Zheng and James, ModalSound, and WaveBlender retain vectors.

**Storage.** Vectors require three times the scalar storage. Langlois et al. 2014 report approximately 100:1 transparent compression of mode-shape fields; a later extension can define compression if needed.

**Implication.** MeshEditor's `mesh2modes` must be upgraded to retain eigenvector 3-vectors (known follow-up).

## Self-contained sample positions

**Decision.** Sample mode shapes at explicit node-local `positions` independent of any mesh. The positions transform rigidly with the node.

Tet-mesh vertices do not generally correspond to render-mesh vertices. Render meshes may split vertices, contain independently indexed primitives, and change through decimation or LOD selection.

Physics contacts provide positions and impulses on collider geometry. Position-based lookup applies without a mesh-index mapping.

Authors control sample placement and density. The representation supports sparse RealImpact-style strike points, dense WaveBlender-style samples, and physically meaningful interior points.

Per-vertex attributes would require one attribute for each of 30–100 modes and retain the mesh-correspondence constraints.

## Nearest-point and barycentric evaluation

**Decision.** φₙ(p) is the value at the nearest sample point; when `indices` supplies a triangulation over the sample points, implementations SHOULD instead project p to the closest point on that triangulation and blend the corner vectors barycentrically.

Nearest-point evaluation requires no topology or more than one kd-tree lookup. Its piecewise-constant field can produce discrete timbre changes with sparse samples. Barycentric interpolation varies timbre continuously, as in van den Doel's “sound map” and WaveBlender's interpolation onto FDTD boundary points. Mode-shape fields are spatially smooth, as also used by Langlois et al. 2014 for compression.

**Requirement level.** Nearest-point converges toward the interpolated result as sample density increases. SHOULD permits minimal implementations to omit closest-point-on-triangle processing.

**Authoring constraint.** Sample spacing must resolve the spatial wavelength of the highest relevant mode, analogous to the FEM *h* < λ/6 rule. Otherwise, interpolation across a sign change underestimates excitation. Lower modes are smoother and usually dominate perception.

## Per-mode decay rates required; Rayleigh material optional

**Decision.** `decayRates` (d, in s⁻¹, envelope e^(−dt)) is a required per-mode array. The acoustic material (ρ, E, ν, α, β) is optional metadata.

FEM models can derive damping from α and β. Measured ACME and RealImpact models fit independent per-mode decays. Stored decay rates represent both sources through one render path. Material data supports damping re-derivation, uniform-scale rescaling, and Hertz contact-duration estimation.

**Damping models.** Rayleigh damping is the special case β₁(A) = α₁A, β₂(A) = α₂A of generalized proportional damping (Adhikari 2006; Sterling and Lin 2016). It cannot fit a different damping-to-frequency exponent. Sterling and Lin's listening test rated their power-law GPD model less realistic than Rayleigh, so the perceptual effect remains unresolved.

A per-mode array represents any fitted distribution. Differentiable modal renderers also learn damping per mode from recordings (Jin et al. 2024).

The Rayleigh form is used at render time only to re-derive damping after uniform rescaling. The rule refers to the material's damping function, so a future general form requires no amendment. An author with fitted decay rates omits α and β, and the fallback retains the measured values.

**Damping curves.** Authoring tools can evaluate fitted curves per mode and store the resulting decay rates. The format omits a separate curve because the evidence does not establish an audible fidelity benefit.

**Units.** Use decay rate *d* in s⁻¹, the continuous-time form used by van den Doel and Pai, JASS, Phya, and RealImpact. The appendix converts T60 = ln(1000)/*d* and Q = π*f*/*d*.

## Frequencies: observed (damped) Hz

**Decision.** `frequencies` stores the observed damped natural frequency in Hz used to tune a resonator.

Hz is sample-rate independent. WaveBlender and openpbso store generalized eigenvalues λ = ω²ρ, which require density to recover Hz. Measured models and renderers use damped frequency directly. Recover the undamped value as ω = √((2π*f*)² + *d*²) for rescaling or damping re-derivation.

## Mass normalization convention

**Decision.** Shapes SHOULD be mass-normalized displacement eigenvectors (ΦᵀMΦ = I, M in kg), giving them units of kg^(−1/2).

Mass-normalized shapes include density, so aₙ = φₙ·j requires no density term and preserves relative levels between models. Spatially varying density remains inside *M*. The openpbso and WaveBlender convention Φᵀ(M/ρ)Φ = I factors out uniform density and requires division by ρ during rendering. The conventions differ by a √ρ factor on φ. Measured models without a mass matrix cannot demonstrate ΦᵀMΦ = I, so SHOULD permits fitted data expressed with equivalent amplitude semantics.

## Mode-major layout + importance ordering

**Decision.** `shapes` is mode-major (mode m's block at m·P + i); modes SHOULD be ordered by decreasing perceptual importance.

Mode-major importance ordering permits a renderer to load a prefix of every accessor, following amplitude-sorted Phya and JASS files. Contact lookup across modes remains strided. SHOULD permits frequency-ordered FEM output while recommending an inexpensive authoring step.

## Excitation semantics: impulse + position

**Decision.** An excitation is (j in N·s, p, t₀), both expressed node-locally: p by the full inverse global transform, j rotated only (physical magnitude preserved). Sustained contacts are impulse sequences, optionally shaped by the contact-force spectrum.

Impulse and position match the `KHR_physics_rigid_bodies` `rigid_body/applyPointImpulse` interface. Contact-event transport remains host-defined. Rotate **j** into local space without scaling its physical magnitude; shape-rescaling rules account for geometry scale. Contact-duration shaping remains SHOULD because deriving Hertz τ requires material data, curvature, and effective mass.

**Force spectrum.** Scale aₙ by F̂(fₙ). A force of duration τ contains little energy above approximately 1/(2τ), and a sampled pulse supplies F̂(fₙ) directly. An output low-pass uses an unrelated cutoff.

**Temporal scope.** A point impulse represents contact duration and its rolloff near 1/(2τ). Finite-patch spatial averaging requires a contact radius absent from the model. Decay rates already represent higher damping in soft materials.

## Normative relative levels

**Decision.** The normative output is the superposition Σ 2πfₙ aₙ e^(−dₙt) sin(2πfₙt) with relative amplitudes across modes/excitations/instances normative and absolute level implementation-defined, and any resonator with matching impulse response is admissible.

This defines cross-implementation amplitude balance while permitting biquad banks, complex one-pole updates, frequency-domain synthesis, and wave solvers. Platforms and mixers control absolute level. The literature uses the specified sine phase, whose perceptual effect is negligible.

## Far-field pressure source signal

**Decision.** The source signal is the far-field pressure a compact source radiates. The modal sum therefore carries a factor of 2πfₙ, and per-mode acoustic transfer data replaces that factor rather than multiplying it. Referring the factor to a fixed frequency is permitted, being a uniform absolute scale.

The modal and acceleration-noise terms must represent the same physical quantity. Multiplying modal displacement by 2πfₙ produces surface velocity, and the compact-source radiation factor converts that velocity to far-field pressure. Omitting the factor changes their normative balance by 2πfₙ, especially for small stiff bodies with ultrasonic modes.

Pressure provides a common output quantity. The dipole derivative removes the unphysical DC component from acceleration noise. The modal term requires the same radiation treatment, including its 6 dB per octave tilt. Roughness-noise measurements also report microphone pressure, enabling direct comparison.

Velocity remains the interface quantity for a radiation solver because the Helmholtz boundary condition is normal velocity. The specification includes the compact-source factor for renderers without transfer data. Renderers with transfer data replace that factor.

Measured against the roughness-noise laws on a sliding contact, the change moves the roughness exponent from 0.82 to 0.69 and the speed exponent from 0.46 to 0.68, which places both in the range direct numerical simulation of sliding rough surfaces reports (Dang et al. 2013) rather than one of the two.

## Pressure level at a stated distance

**Decision.** Implementations that provide physical levels state the output quantity and reference distance *r*₀. They render far-field pressure in pascals at that distance and attenuate each body by *r*₀/max(*d*, *r*₀). Audibility culling uses the *r*₀ level.

**Reference.** A common physical quantity and distance make levels comparable across assets, objects, and mechanisms.

**Attenuation.** A physical output level requires a distance reference. The clamp inside *r*₀ bounds output for a listener passing through a body. Directivity, occlusion, and reverberation remain outside this extension.

**Culling.** Determine the active modes at *r*₀ and apply listener attenuation only to the mixed output. This keeps modal decay independent of listener motion.

## Acceleration noise in scope, mass properties self-contained

**Decision.** Rigid-body acceleration noise (the contact "click") is a SHOULD-level render feature driven by the same excitation as the modes, radiated omnidirectionally. Mass, center of mass, and inertia live in an optional `massProperties` block mirroring `KHR_physics_rigid_bodies`. When absent they MAY come from that physics extension or from watertight geometry plus ρ, with the model's own values authoritative.

**Why.** Acceleration noise is the primary sound for small stiff bodies with ultrasonic modes. Its analytic form (Δv = j/M, Δω = I⁻¹(r×j), half-sine force pulse, dipole far field ∝ ρ₀V·ȧ) requires no precomputed radiation data. A compact recoiling body has no monopole, so its leading radiation is a dipole whose pressure uses the derivative of acceleration (Curle 1955, Morse & Ingard). The derivative also removes the raw acceleration pulse's unphysical DC component. Self-contained mass properties preserve audio when the physics extension is absent and allow acoustic mass to differ from gameplay mass. Omnidirectional rendering matches the modal-core approximation; directivity remains deferred.

**Impulsive excitation only.** A draft carried the same mechanism over to sustained contact, with **F**(*t*) in place of the impulse pulse. That is derivable, since the dipole result assumes only a compact body under a net force and the net force on a loaded body is the fluctuation the excitation already is, but it is not how the field models sliding noise and it was withdrawn. Le Bot 2017 projects the vibration onto a modal basis driven by modal contact forces, with no rigid-body term, and his own simulation slides a steel cube on an elastic plate and radiates from the plate. A body in sustained contact is also coupled to the counterface through the contact stiffness rather than free, so its mass and that stiffness form a resonator. Radiation from recoil absent from the contact model would represent the wrong side of that resonator. The mechanism may be negligible or unstudied for sliding; the specification makes no claim.

## Nyquist culling

**Decision.** Implementations MUST NOT produce aliased output from modes at or above the output Nyquist frequency; equivalently, such modes contribute no output.

**Why MUST.** A resonator above Nyquist aliases to fs − f and produces a nonphysical partial. Implementations may skip the mode or use a higher internal rate with anti-aliased decimation, as in WaveBlender. The requirement ensures that lower output rates produce a low-passed result without added inharmonic tones.

## Scale semantics

**Decision.** Modal data describes the object at its world size in the scene's *initial* state. Uniform rescale by γ: implementations SHOULD apply ω → ω/γ, φ → γ^(−3/2)φ, re-derive d from material α/β when present (the βω² term is scale-dependent), then f from (ω, d). Non-uniform scale: undefined.

Frequency scales inversely with uniform size. The exact laws come from Zheng and James 2010, Appendix E. SHOULD permits a minimal renderer to retain stored data. Non-uniform scaling changes mode shapes and cannot be reconstructed from stored modes, following the `KHR_gaussian_splatting` precedent for unsupported transforms. The initial pre-animation state gives loaders a detectable reference.

## One model instance per GPU render instance

**Decision.** A node with `EXT_mesh_gpu_instancing` and a modal model instantiates the model once per render instance, each with independent oscillator state and the node's `gain`. Excitation attribution to a render instance is host logic (same scoping as contact reporting); once attributed, all excitation math uses the composed transform (node global × instance TRS), and each instance is a source at its own origin. The model is referenced to identity instance transform, so a per-instance uniform `SCALE` is handled by the standard scale-adjustment rules; non-uniform composed scale is undefined.

Independent state lets one struck instance sound without affecting its peers. Uniform instance `SCALE` applies the runtime-rescaling technique of Zheng and James 2010 to a shared model. The host attributes excitations because the physics layer does not define per-GPU-instance colliders.

## Sustained-contact excitation

**Decision.** Render sliding, scraping, and rolling as continuous forces driving the modes, mode shapes, and mass properties used for impacts.

Each body has a sweep speed: the rate at which the contact moves over that body's surface. Slip is the difference between the sweeps. Pure rolling has equal sweeps and zero slip. A box sliding on a fixed floor has zero sweep on the box and floor sweep equal to slip. Partial slip varies continuously between these limits.

**Sweep is per body.** A sliding box has zero sweep on its own face while the floor passes through the contact. Collapsing both surfaces into one sweep speed can silence this case. Each surface therefore uses its own sweep rate and contributes separately. A single synthesized track may use the faster sweep only when the other surface is stationary relative to the contact. Separate rolling and sliding models would introduce an audible transition across a continuous physical state. Rigid-body solvers already provide both speeds.

**Excitation definition.** The impulse form is the zero-duration limit of the sustained form. A renderer using sampled force pulses can substitute the sustained force signal without changing the modal response.

## Contact-force requirements

**Decision.** The spec states properties any conformant contact force must have (non-negative, silent at rest, traversal indexed by distance, √v·N amplitude, roughness low-passed by the region that carries the load, force limited far above the load) and puts one satisfying model in a non-normative appendix.

**Why the clamp is a MUST.** Perret-Liaudet and Rigaud 2003 compare Hertzian and linear force laws with the same contact-loss clamp. Their responses match near the primary peak and differ only at higher frequencies. The separation nonlinearity controls the main audible behavior, while implementations may choose the force law.

**Why the per-region clamp is a SHOULD.** A rough contact distributes its load across regions that release and re-engage at different moments. Clamping only the aggregate force averages out this intermittency. Per-region clamping describes a decomposition rather than the total force, so it remains implementation guidance. The section reserves MUST requirements for observable output: non-negative force, equal and opposite excitation, and silence at rest.

**Why the reference bed stays in the appendix.** Rendering the contact as a bed of Hertz spots is one way to satisfy the clamp guidance, and Appendix B carries it as such. Naming it normatively would fix an implementation where the extension only needs an audible result, and would exclude models that reach the same behaviour by resolving the interface directly.

**The limit sits far above the load, not at it.** A sliding rough contact meets its counterface as a succession of micro-impacts whose peaks reach many times the mean load, which is the premise of the roughness-noise literature (Grégoire et al. 2021 measure the individual events). A limit at the load compresses that fluctuation rather than bounding it, and it is the roughness itself being compressed: sweeping the knee from one times the load to a hundred moves the measured roughness exponent from 0.26 to 0.82 against a literature range of 0.7 to 0.96. A limit is still needed, because both force laws grow without bound in approach and a contact whose geometry changes under it can report an approach from far outside the regime either law describes.

**Why.** This matches how [Synthesis](#synthesis-normative-relative-levels-free-implementation) is already scoped: pin the perceptible behavior, leave the architecture free. Mandating Hunt and Crossley would exclude equally valid formulations, and mandating nothing would permit fixed-rate noise that neither tracks speed nor stops, a qualitative error.

**Two requirements are MUST.** Non-negative force preserves the separation nonlinearity that produces micro-collisions and chatter. Silence at rest excludes the equilibrium Hertz force from excitation.

**Parametric surfaces are statistical, stored profiles are reproducible.** A surface given only by roughness, correlation length, and spectral slope specifies an ensemble, so two renderers agree on character but not sample-for-sample. The optional `profile` accessor is the escape hatch, and it also matches the format's philosophy of shipping baked results rather than standardizing a generator (the MPEG-4 Structured Audio anti-pattern noted below). Measured tracks are small: a few thousand samples at a few microns covers enough surface to loop inaudibly.

**The spectral slope is the one-dimensional exponent, and the fractal relation is self-affine.** *p* describes the profile stored in `profile` and traversed by a contact. For a self-affine surface the two-dimensional power spectrum scales as *q*^−2−2*H* and a line scan scales as *q*^−1−2*H* (Jacobs et al. 2017 Eq. 48). Inverting the one-dimensional form gives *H* = −(1 + *p*)/2. The surface fractal dimension is 3 − *H* (Persson et al. 2005 Appendix B), and the profile dimension is 2 − *H* = *p*/2 + 2.5. The specification gives both *H*, used in contact mechanics, and profile *D*, used in surface metrology.

van den Doel et al. 2001 state *D* = *p*/2 + 2 for the same profile quantity, following a phonograph-needle model whose wavetable is a height track exactly as `profile` is. That form is half a unit off and changes the presets: the machined row's measured *D* = 1.3 becomes *p* = −1.4 and so *H* = 0.2, against the 0.7 to 0.9 that spectral analysis reports for surfaces from atomic to geological scales (Jacobs et al. 2017) and the 0.8 that contact-mechanics work takes as the common case while calling lower values rare (Papangelo et al. 2017). The two forms are easiest to separate at *p* = −2, where the self-affine relation gives the Brownian value *D* = 1.5 and van den Doel's gives *D* = 1, a smooth rectifiable line, which a nowhere-differentiable Brownian profile is not. The measured fractal dimensions are the anchor, so the table keeps its *D* column and recomputes *p* = 2*D* − 5.

**The *D* column's spread is wider than measurement supports.** Only the machined row is anchored. Recomputing *p* from the illustrative values gives *H* = 0.9 for polished, 0.6 for sandblasted, and 0.5 for cast. The last two fall below the reported range. Measured fractal dimensions per finish would resolve that gap. The table distinguishes finishes primarily through σ and *ℓ*, which span three orders of magnitude.

## Real contact area across regimes

**Decision.** The area the asperities touch over comes from a single expression, *A* = *A*₀ erf(√π *N*/(2 *A*₀ *p*)) with *p* = *E*\*|∇*h*|<sub>rms</sub>/2 and *A*₀ the smaller of the Hertz patch and the shared polygon (Pastewka and Robbins 2016). The microscale finish is filtered over the width of *A* and longer relief over the width of *A*₀. The force law still distinguishes a load-set contact from a geometry-set one, and the area does not.

**Why one expression rather than two cases.** Its limits are the two cases. At light load the error function linearizes to *A* = *N*/*p*, independent of *A*₀, which is the nominally flat rough result. At heavy load it saturates at *A*₀, which is the Hertz result. Between them it interpolates with no free parameter and no threshold, and it was validated against molecular simulations of rough spheres from 30 nm to 30 µm over ten orders of magnitude in load. A case split on geometry would have to answer a question the geometry cannot: the physical crossover is set by *h*<sub>rms</sub>/δ, the roughness over the smooth-surface penetration, which is Hertz-like below 0.01 and roughness-dominated above 10 (Tiwari and Persson 2020). Ordinary objects span that whole range, so a curved contact on a rough surface sits in the middle while a geometric test would still send it to Hertz.

**Why the finish and the relief are filtered over different widths.** The two widths differ by orders of magnitude, so one of them cannot serve both scales. Filtering the microscale finish over the confining region instead attenuates it to nothing wherever two faces meet, since that region is the whole shared polygon while the finish's own contacts span microns.

**Greenwood and Tripp comparison.** Their rough-sphere treatment uses the Greenwood and Williamson model of noninteracting, equal-radius asperities. Müser et al. 2017 report unreliable interfacial separation and contact-patch-size distributions for this model family. Those quantities directly affect contact sound.

**Asperity pressure.** *p* = *E*\*|∇*h*|<sub>rms</sub>/2 is independent of load and area, producing real area proportional to load at light load. Persson's theory gives the expression analytically, and the numerical studies cited by Pastewka and Robbins give the same coefficient within their reported accuracy.

**The force law is no longer branched either.** Two faces bear on the asperities standing proud inside the region confining them, so a geometry-set contact is a bed of Hertz spots and each spot takes the load-set law at its own radius. Persson's *N* exp(δ/*u*₀) is the *mean* of such a bed, and a mean is the wrong object here: the sound is made by individual spots parting and re-engaging, which averaging removes. The exponential's separation scale *u*₀ survives as the scale the bed's mean force decays over and as a check on its stiffness.

## Asperity radius at the finest surface scale

**Decision.** Where an implementation renders the contact as a bed of discrete Hertz spots, each spot's radius of curvature is the one the surface carries at the shortest wavelength it resolves, ρ = 2⟨|∇²*h*|²⟩^(−1/2) (Pastewka and Robbins 2016). It is not coarsened to the width of the spot bearing on it.

**Why.** Molecular simulations validate this radius in the low-load, first-asperity limit. It also reconciles the bed with [the area law](#real-contact-area-across-regimes): summing Hertz-spot areas over the height distribution reproduces the error-function area within a few percent. Coarsening the radius to the spot width breaks that agreement because spot area depends on radius while the area law is curvature-independent.

**Model limitation.** Müser et al. 2017 observe that rms curvature is set by the finest scale while most contact patches are much larger. They also report that bearing models overestimate typical patch size by a factor of ten, so increasing the spot radius would increase the error. Real patches merge through long-range elasticity between peaks; a single-radius asperity model cannot represent that geometry.

**Equivalent conventions.** The equation gives ρ as 2⟨|∇²*h*|²⟩^(−1/2) for the two-dimensional Laplacian and as λ<sub>s</sub>/(2π|∇*h*|<sub>rms</sub>) √(2(2−*H*)/(1−*H*)) in surface parameters. The factor of two converts between a profile's second derivative and the Laplacian, so a one-dimensional curvature measurement uses no additional factor. The second form requires no moment conversion. The forms agree across the self-affine range and diverge as *H* approaches one, where √(1/(1−*H*)) diverges while sampled second differences remain fixed at the sampling scale.

**Revisit when** the bed carries a patch-size distribution rather than a population of equal spots. The radius then describes a patch rather than a summit, and the scale it is read at follows.

## Meter-scaled element ribbon and load-bearing strip count

**Decision.** The elements a sustained contact bears on tile a ribbon of the interface whose extents are lengths rather than sample counts. Along the sweep the ribbon runs the distance a slide covers before the surface repeats, and across the sweep it samples one strip four correlation lengths wide. The width a contact actually bears over is that ribbon's area divided by its length, and the spring count represents the number of strips across that width.

**Why.** A sample count standing in for a length makes the physics track the sampling. The field is synthesized at the spacing the surface's own short-wavelength cutoff sets, so a fixed count of columns and rows covers less ground every time the band is described more finely: the same contact then bears over less area, seats deeper for it, and reports every bearing statistic at a contact the pair is not making. Pastewka and Robbins 2016 state their own domain the same way this decision does, as a band between two wavelengths plus a padding region that keeps the periodic images apart, with the discretization following.

**Strip width.** Sampling the across-sweep direction avoids resolving the finest wavelength over a footprint tens of millimetres wide. Pastewka et al. 2013 Eq. (B3) gives the captured height-variance fraction as (1 + *H*(1 − (*l*/*L*)²)) / (1 + *H*). At *L* = 4*l* and *H* = 0.7, the fraction is 0.987. `SPRING_STRIP_CORRELATIONS` defines this width.

**Measured effect.** PressedRing/b_Grip uses 272 strips and divides the load by 272. Tilt changes from 0.403 to 0.512, flank stiffness from 1.6e6 to 7.9e6 N/m, and the seated bench's top-rung exponent from *n* = 2.49 to 2.59, within the measured 2.5–3 microslip band. Across fourfold band refinement, summit density is 2.28, 2.56, and 2.64 × 10¹⁰/m²; mean summit stiffness is 4.13, 4.00, and 3.97 × 10⁷; and mean element crest is 7.52, 7.62, and 7.61 μm. Strip and bearing widths remain constant.

**Bearing population.** Four correlation lengths capture the strip spectrum but contain too few bearing asperities. Pastewka and Robbins 2016 Eq. (5), using asperity radius in place of sphere radius, gives the first-asperity threshold *N*<sub>c</sub> = 0.092 N for this pair. The full contact carries 4.90 N, while one of 272 strips carries 0.018 N, one fifth of *N*<sub>c</sub>.

Measured across four strip widths at three surface realizations each, the flank moment falls steeply while the ribbon is below Nc and goes flat once it is above: 16.9, 6.23, 4.03, 4.18 × 10¹³ at 4, 16, 32 and 64 correlation lengths, whose ribbon loads are 0.20, 0.78, 1.56 and 3.1 times Nc. The mean bearing width flattens with it, 7.12, 9.53, 10.26, 10.24 × 10⁻⁷ m. So Eq. (5) predicts the threshold at twenty correlation lengths and convergence is observed at thirty-two.

**Strip aggregation.** Widening one strip increases memory quadratically; *N*/*N*<sub>c</sub> ≈ 10 would require an 11 mm strip and a multi-gigabyte field. Instead, synthesize eight independent strips sequentially and retain each element's highest summit. Only the summits persist between realizations.

**Measured effect.** Eight four-correlation-length strips produce flank moment 3.67 × 10¹³, compared with 4.03 for one 32-length strip, 4.18 for one 64-length strip, and 16.9 for one four-length strip. Peak memory is 131 MB instead of 1.05 GB. `SPRING_STRIP_REALIZATIONS` counts strips and `SPRING_STRIP_CORRELATIONS` sizes each strip.

**Revisit when.** Limit gathered strips when a contact's transverse extent contains fewer strips than the ribbon. The strip count already has a minimum of one, but the current design does not verify that the gathered ribbon fits inside the represented footprint.

## Measured-profile element synthesis

**Decision.** A surface carrying a measured height track builds its element springs from a field that track is a row of. The field's radial spectrum is read from the track's own by inverting the integral a cut takes over the perpendicular wavenumber, the track's heights land on the field exactly by conditioning the draw on them, and the field spans the length the track was measured over at the sampling it was measured at. So a measured surface repeats where its measurement ends rather than at the synthesized ribbon's own sweep budget, and every other band the pair carries is drawn on that same grid. The correlation length and spectral slope such a surface needs are fitted to the track rather than read from the parameters beside it.

**Why.** The elements tile a two-dimensional field and a measurement is one trace, so the two do not meet without a construction. Leaving them apart is not a limit a measured surface has to live with. It is a gap: a surface carrying a track silently keeps the statistical bed however the container is flagged, which makes the measured arm of a measured-versus-synthesized comparison a bed arm. It also blocks the bed's deletion outright, since the bed would be the only path a measured surface has.

**Spectrum input.** The stored profile provides a measured surface realization. Replacing it with a fitted corner and slope would discard that measurement. The fitted band remains necessary to determine element width and extend sub-contact roughness below the measured band. Derive those values from the profile spectrum rather than storing duplicate parameters.

**Regularization.** A cut of an isotropic field integrates radial-spectrum values at equal or greater wavenumber. At *q*, its own bin contributes approximately 2 d*q*<sub>y</sub>/(π*q*): one thirteenth near the middle and one twenty-fifth near the top of a machined-finish band on a one-correlation-length strip. Exact inversion amplifies trace scatter by the reciprocal share; measured error was 44–96% across the decaying band. Regularize this Abel-like inversion on one-sixteenth-octave cells. Seed it with the one-power-shallower parametric relation and refine multiplicatively against the reconstructed cut.

**Measured effect.** Field cuts reproduce the trace spectrum within 1% over the decaying band, with a slope ratio of 1.006. Recovered row heights differ by at most 1e-4 rms height. On the same surface and grid, the parametric patch is 24–37% low across the band, 14% shallower in slope, and 22–35% high on the plateau. Discrete perpendicular sampling raises the plateau. A radial cutoff truncates the integration arc and rolls off a cut before the profile cutoff. Both methods retain the low-end plateau bias from a strip only four correlation lengths wide.

**Revisit when.** Apply the same refinement to parametric spectra if correcting their 24–37% error justifies replacing a closed form with an inversion.

## Oblique-flank moment over bearing contacts

**Decision.** The moment that sizes oblique-flank micro-slip is Σ slope²/force over the summits carrying load, tabulated against engagement alongside each element's force curve, and the width the flank tilt is read at divides the bearing stiffness by that same count of summits. An element is not a contact.

**Why.** The loop area a tilted flank sheds per cycle is (4/3)*k*<sub>t</sub>²(*a* sin θ)³/(μ*F*) for one contact, so the population's moment is a sum over contacts. An element one quarter of a correlation length wide contains tens of summits, and several can bear at once. (Σ slope)² / (Σ force) equals Σ slope²/force only where every bearing summit sits at one depth. Otherwise it is smaller by Cauchy-Schwarz, and the gap grows with the element's depth spread, making the moment dependent on the tiling. The same argument applies to width: d*F*/d*d* = 2*aE** applies to one axisymmetric contact, so dividing aggregate element stiffness by the element count measures the element rather than the contact.

**Measured effect.** The forms agree when one summit bears per element, as in one of four PressedRing surface realizations at its own pressure. Across all four realizations, the moment increases by up to 1.5×. On the seated bench, the top-rung amplitude exponent changes from *n* = 2.59 to 2.71 with a 76% flank share, within the measured 2.5–3 microslip band.

## Flank tilt at bearing-patch width

**Decision.** The tilt that oblique-flank micro-slip slides against is the surface's rms gradient low-passed at the width of the contact its bearing asperities carry the load on, 2*a* for *a* = √(*Rd*) at the Hertz depth *d* = 1.5*N*/*K*. It is not the gradient at the shortest wavelength the surface resolves.

**Why.** Pastewka et al. 2013 Appendix B splits a contact's own shape from the roughness inside it at *q* = π/*r*<sub>0</sub> for a contact of radius *r*<sub>0</sub>: their Eq. (B6) builds the mesoasperity's curvature from components below that wavevector and their Eq. (B18) leaves everything above it as roughness within the contact. A wavelength shorter than the patch does not tilt the patch. It is relief the patch sits across, and it forms its own contacts one level down the hierarchy, which a model carrying a single level of elements does not represent. Müser et al. 2017 section 4.2 reaches the same limit from the measured side, that the finest scale sets local quantities such as rms gradient while real contact points belong to patches far larger.

**Scale distinction.** Radius controls the growth of a summit contact under load and is validated in the first-asperity limit. Tilt is the average material-plane orientation over the existing patch. Increasing radius would inflate contact area; patch-scale tilt does not change area.

**Measured effect.** Element width follows d*F*/d*d* = 2*aE**, and the authored short wavelength fixes the tilt scale. PressedRing/b_Grip bearing contacts are narrower than profile spacing at the contact's current pressure, so they use the finest-scale tilt. At the previous pressure, 272 times higher, patch-scale tilt was 0.41 versus a finest-scale value of 0.51.

**And a wider band cannot supply it.** Combining Hertz at one summit, the linear area law, and Pastewka & Robbins Eq. (6) for the summit radius, the load and the surface gradient both cancel and leave the bearing contact's width as 2*a* = (3/16)√(2(2−*H*)/(1−*H*)) λ<sub>s</sub>, which is 1.10 λ<sub>s</sub> at *H* = 0.7. The bearing contact sits at about one short-wavelength cutoff whatever the load, so authoring a finer band sharpens the summits at exactly the rate that shrinks their contacts. No authored band ever contains its own contacts.

Nor is a finer band authorable. ISO 3274's short-wavelength cutoff is a property of the measuring instrument, set by the stylus tip radius, rather than of the surface, and ISO 4288 puts a machined finish of Ra 0.1 to 2 μm on a 0.8 mm profile filter with a 2 μm tip, which resolves nothing finer than a few microns (mitutoyo_surface_roughness and willrich_surface_parameters, held in papers_surface_metrology.tar.gz, restate those tables rather than being the standard). So a surface authored from a measurement carries no roughness at the scale its own contacts form at.

## Sub-contact roughness compliance

**Decision.** Divide every summit's depth between its Hertz contact and the sub-cutoff surface band inside that contact. For contact depth *x* and width *w* = 1.5*k*√*x*/*E**, the interface displacement is (3/*H*) γ *A* *w*<sup>*H*</sup>, where γ *A* *w*<sup>*H*</sup> is the band's mean separation. The two compliances add in series per summit, and the summit contributions add across the population.

**Why.** The previous entry proves a band can never contain its own contacts, so the level below is not reachable by describing the surface more finely and has to be carried as a term. Pastewka et al. 2013 Appendix B gives that term as a second elastic energy of the same power in load as the contacts' own, combining with the first as 1/θ = 1/θ<sub>0</sub> + 1/θ<sub>1</sub> (their Eq. B27), and its prefactor is γ, β and *H* alone. Inside the patch the interface follows Persson's law at the patch's own pressure, so its stiffness is the load over the separation, and integrating d*F*/d(depth) = *F*/*u* along the load leaves (3/*H*) *u* of depth. Both depths are then powers of the load, which is why the whole population shares one curve and a summit enters it through one scale of its own.

**The incremental law is the primitive, not the energy.** Appendix B writes *U*<sup>(1)</sup> = *u*<sub>1</sub>*F* as an evaluation at the load, not as the work integral of a separation that varies along the path. Taking it as the integral instead gives (3+*H*)/*H* rather than (3/*H*), a fifth too much depth. Persson's K = *p*/*u*<sub>0</sub>, restated by the appendix as Eq. (B28), applies pointwise.

**It cannot be aggregated.** These springs sit in series with each summit while the summits sit in parallel, so the stack is Σ<sub>i</sub>(1/*k*<sub>i</sub> + *u*<sub>i</sub>/*F*<sub>i</sub>)<sup>−1</sup> and no single series element at the anchor reproduces it. The separation also stops widening at the stated cutoff: a contact wider than that has the field's own resolved summits standing inside it, and reading the band at its full width would count them twice.

**Measured effect.** On PressedRing/b_Grip at 4.90 N over a fourfold short-wavelength span, interface compliance changes 23% with this term and 43% without it. At the shipping band, bearing contacts narrow from 0.99 to 0.68 μm, nearly twice as many contacts bear, and interface stiffness remains nearly constant.

**Remaining approximation.** Appendix Eq. (B6) integrates the spectrum to π/*r*<sub>0</sub>, where *r*<sub>0</sub> is the contact width. The implementation instead uses authored-band curvature, so the authored cutoff still affects summit sharpness. A self-consistent solution must derive width and radius together and use contact width as the cutoff. For *H* < 1, the upper limit dominates ∫*q*<sup>5</sup>*C*(*q*)d*q*, so convergence requires a physical length.

## Contact stiffness capped by the shared polygon

**Decision.** *k* = 2*E*\* *a* with *a* = min(√(*R*\*δ), √(*A*₀<sub>poly</sub>/π)), and the contact time is the collision against that one law, integrated. Hertz's τ = 2.868 (*m*\*²/(*E*\*² *R*\* *v*))^(1/5) and the flat punch's τ = π √(*m*\*/*k*) are its limits, not its cases.

**Why this is a unification rather than an interpolation.** 2*E*\* *a* is exact at both ends. Differentiating Hertz's own *f*<sub>n</sub> = (4/3) *E*\*√*R*\* δ^(3/2) gives d*f*<sub>n</sub>/dδ = 2*E*\*√(*R*\*δ), which is 2*E*\* *a* at the Hertz patch radius, and a flat punch of radius *a* on an elastic half space has stiffness 2*E*\* *a* outright. The two agree in value at the depth the patch fills, so the capped form is continuous with no constant fitted between them. The speed dependence then follows rather than being asserted: a power-law spring *f* ∝ δ^p gives τ ∝ *v*^((p−1)/(p+1)), which is *v*^(−1/5) while the patch grows and nothing once it has filled.

**Shared patch definition.** π*a*(δ)² is the bounding region *A*₀ that [the real contact area](#real-contact-area-across-regimes) defines as the smaller of the Hertz patch and shared polygon. The stiffness and area laws use the same patch.

**Why not branch on whether the physics reports an area.** That question is answered by the solver, not by the contact. Whether a manifold comes back with three points or two depends on tessellation and on how exactly the bodies met, so a body landing a degree off level would cross the branch and its contact time would step. The cap crosses smoothly in both directions: a polygon wider than the patch ever grows changes nothing, and one narrower takes over as soon as the patch reaches it.

**Limitations.** The rigid punch on a smooth half-space represents the pair's highest stiffness. Real interfaces first load asperities and therefore produce a softer, longer contact. Interfacial stiffness should act in series with bulk stiffness. Edge and corner contacts have singular curvature and remain approximated by the curvature floor.

## Reusable contact description

**Decision.** A contact is described once, by its geometry, load, materials, and roughness, and the force follows from that description in every regime. The contact zone resolves into springs distributed across it, each bearing only where the surface reaches it, so no branch selects between a curved-contact model and a flat-contact model, and no rule decides when a contact stops being an impact and starts being a scrape.

**Shared model.** Separate smooth-Hertz impact and fractal-track scraping models require normative switching rules and a third rolling model. One contact-mechanics description covers all three through the geometric term and patch-filtered relief. [Vibrational coupling](#vibrational-coupling) also requires instantaneous contact area, including between discrete events.

**Discrete springs.** Thompson et al. 2003 report excessive high-frequency attenuation from an analytical spatial-average contact filter compared with distributed springs driven by measured roughness. Independently loaded springs cover the fully distributed and single-contact limits as load changes. The same evaluation supplies separation nonlinearity and instantaneous contact area for [vibrational coupling](#vibrational-coupling).

**Which limit a contact sits in is a number, not a choice.** Hertz idealizes roughness away and Persson idealizes curvature away, so each diverges exactly where the other applies, at *R*\* → ∞ and at *A*₀ → 0. Two ratios decide which applies: *h*<sub>rms</sub> over the smooth-surface penetration, and the count of asperities bearing load. Both follow from quantities a contact already has, and across ordinary objects both span several decades, so a renderer that commits to one regime is wrong somewhere unremarkable.

**Limitations.** Independent springs omit elastic coupling, including effects reported at light load; restoring it requires a coupled solve per step. Elastic asperities approximate hard ceramics and soft polymers but fail when machined-steel contact pressure exceeds hardness. A one-dimensional profile samples bearing asperities from a line instead of an area.

## Contact curvature comes from the object's own geometry

**Decision.** κ is the mean surface curvature at the contact point, read from the mesh of the collider the contact landed on, or from that of its nearest ancestor with a mesh. The extension carries no curvature data, so an implementation derives it.

**Collision-geometry limitation.** Common collision proxies omit render-surface curvature. A convex hull replaces curvature with facets, while a primitive uses an analytic shape that may not fit the render surface. Flat curvature can understate κ without bound. Through *R*\* = 1/(κ₁ + κ₂), *k* ∝ √*R*\*, and *a* ∝ *R*\*^(1/3), the error affects stiffness, patch radius, and contact time. A ceramic bowl represented by a collision hull would receive the contact time of a flat plate.

**Where the two coincide.** A body whose collision shape is its authored shape has one geometry, and its analytic curvature is exact. The distinction bites only when a collider stands in for something finer.

**Why geometry resolves on its own walk.** Curvature is a fact about the shape under the contact, while an acoustic surface is authored finish. Resolving κ through the surface walk would make it depend on where an author chose to attach a finish, so a compound whose foot carries a mesh but no surface would read the curvature of the body's mesh instead of the foot's. The two walks look alike and answer different questions, so they are stated separately.

**Cost accepted.** Deriving κ at the contact requires locating the contact on the body's surface geometry, which is unavailable to a renderer with only collision proxies. Mesoscale relief requires the same capability to sample `normalTexture` at **p**.

**Mean curvature is the axisymmetric form.** Johnson's general solution writes the gap between two surfaces as *Ax*² + *By*², where *A* + *B* is half the sum of all four principal curvatures and equals κ₁ + κ₂ for mean curvatures. Where the contact is axisymmetric the patch is a circle and that sum is the whole solution. Where the principal curvatures differ the patch is an ellipse whose shape also depends on *B* − *A*, which mean curvature does not carry. The sum term is kept because it is exact for the cases the model is most often applied to and needs one number per body rather than a curvature tensor and a relative orientation.

**Shared-polygon sensitivity.** A curved convex body's exact collision hull is faceted, so a contact can report real curvature and a facet polygon together. [The patch radius](#contact-stiffness-capped-by-the-shared-polygon) uses the smaller constraint. Curvature controls light loads whose Hertz patch is narrower than a facet; the facet controls loads broad enough to bear over it. Using only the polygon would assign a hulled sphere the contact time of a small flat punch.

## One modal model per rigid body and one surface per collider

**Decision.** A contact resolves both the modal model and the acoustic surface from the collider node it touched, or its nearest ancestor carrying one. At most one node of a rigid body's hierarchy may carry a model. Any number may carry a surface.

**Why the walk starts at the collider.** A machine on rubber feet with a steel shell is one body, and a contact on a foot must find rubber where one on the shell finds steel. Finish, elastic constants, and curvature all belong to the geometry touched.

**Why only one model.** Modes are the eigen-decomposition of one connected structure and span every sub-shape of a compound. A hammer's head and handle are rigidly joined, so striking the head drives modes reaching through the handle. A collider hierarchy decomposes collision geometry, not the vibrating object. Models per collider would give one body several uncoupled resonators and make its spectrum depend on which part was struck.

**Rigid-body scope.** Nesting does not define elasticity: cups parented to a shelf require separate models. A rigid body asserts that its parts move together.

## One body, one mass: shapes renormalize to the authoritative mass at fixed frequencies

**Decision.** Every mass-consuming path (contact dynamics, Hertz contact time, acceleration noise, and the modal shapes' normalization) reads the body's one authoritative mass. When a model's shapes were derived against a different solve mass, they scale by √(*M*ₛ/*M*) and the frequencies stay.

**Why one mass.** A body's authored mass and the solid mass its geometry implies at the material's density can disagree by orders of magnitude, as when a physics demo authors a large box at a nominal 1 kg. The dynamics, the recoil, the contact time, and the shapes' normalization each consume a mass, and relative loudness across those paths is meaningful only when they all read the same one.

**Why frequencies stay.** Scaling shapes at fixed frequencies preserves the material's frequency character while making the struck response carry the energy an impulse delivers to a body of mass *M*. It is exactly the solve of a body whose density and stiffness both scale by *M*/*M*ₛ, a lighter material at the same specific stiffness *E*/ρ. Specific stiffness varies far less across real materials than density does, so this is the least-wrong homogeneous reading of "this geometry, this material character, that mass". It is also the first-order account of a hollow body of the material: a shell rings in a similar band to the solid, responds harder per impulse, and keeps the solid's surface hardness, which is why the contact time keeps the material's true *E*.

**Why not rescale density at fixed stiffness.** That reading shifts every frequency by √(*M*ₛ/*M*). The 1 kg box's spectrum would move up 27 fold, mostly past hearing, and the implied material (ceramic stiffness at balloon density) exists nowhere. It also silences exactly the bodies whose response should grow.

**Shell limitation.** A shell or composite eigensolve gives the exact modes for a body lighter than its solid equivalent. The asset does not locate material, and mass alone can imply an implausible wall thickness. The renormalized solid is a bounded first-order approximation. Authors needing shell acoustics can provide a shell-derived modal model.

**Energy consistency.** With shapes normalized to *M*, an impulse *j* deposits modal energy on the scale of *j*²/2*M*, the same budget the rigid recoil draws from.

## Absolute surface scales and separate materials

**Decision.** Acoustic surfaces are their own array, not fields on the acoustic material, and their lengths are absolute physical quantities exempt from node scaling.

**Why separate.** Bulk and finish vary independently: polished and sandblasted steel share every material constant and sound completely different when scraped. Merging them would duplicate five material constants per finish and, worse, create a precedence question when a node's surface and its model's material disagree. A surface referencing a material has no such conflict, and gives a silent floor a way to carry the elastic constants that contact stiffness needs.

**Why exempt from scaling.** Modal data describes an object at one physical size and rescales with it, but a finish does not: a scaled-up polished sphere is still polished. Contact position and sweep speed are geometric and do transform, and they keep their magnitudes when a contact state is expressed in node-local space, since a node-scaled velocity would read an absolute finish at the wrong rate. The exemption covers the microscale parameters only. Mesoscale relief is bound to texture coordinates rather than to an absolute length, so it scales with the node like the geometry it is painted on, and a scaled-up tiled floor has larger tiles to both eye and ear.

## Two surface scales, with the mesoscale bound to mesh UVs

**Decision.** A surface carries statistical microscale parameters plus an optional `normalTexture` for mesoscale relief, interpreted as glTF core's `normalTextureInfo` and sampled along the contact path. Absent, it falls back to the contacted primitive's material `normalTexture`, so the surface property overrides the correspondence rather than establishing it.

**Fallback.** Most assets should reuse the material's `normalTexture`. A fallback keeps texture indices synchronized when a material changes and applies immediately to the 55 of 148 Khronos sample models that declare a normal map. Inherit the full `normalTextureInfo`, including `scale`, because texture values and scale jointly define the visible gradient. Partial inheritance could give audible and visible relief different amplitudes.

**Per-primitive resolution.** A glTF mesh uses primitives for regions with different materials. Texture coordinate sets are also declared per primitive, so a `texCoord` index from another primitive may be absent. Resolve the material and coordinates from the contacted primitive.

**Why two scales.** Ren et al. 2010 decompose contact surfaces into macro geometry, mesoscale bumpiness carried by normal maps, and microscale roughness, and state the gap directly: fractal noise alone "does not render any information for the bumpiness or heterogeneous variation of the contacting geometry at the meso level," which is "clearly visible to the users but transparent to the rigid-body simulator." Tiling, grout, corrugation, knurling, and grain live in exactly that band. The collision mesh is too coarse for them and a statistical finish is orders of magnitude too fine.

**Why it matters most for rolling.** The contact patch filter selects between the scales. A 1 cm steel ball on steel under 1 N has a Hertz radius near 40 µm, so a micron-scale finish sits at or below the patch and is strongly attenuated, while millimetre-scale relief passes intact. Rolling hears the tiles and not the polish. Ren's own mechanism is a normal-direction impulse, which is precisely the channel that survives at zero slip, so the mesoscale layer strengthens rolling more than scraping.

**UV binding.** Modal data has no required correspondence with render topology. Surface relief is already a rendered, UV-mapped property. Reusing the material normal map aligns audible and visible spatial variation.

**Why this was available and unused.** Ren's meso contribution did not propagate. Later work in this line cites that paper for its source-filter architecture and for fractal-noise friction alongside van den Doel, never for the mesoscale layer. The idea needs a graphics pipeline to read a normal map from, which perceptual research using measured depth profiles does not have. glTF is the carrier it was missing.

**Consequence for scope.** Spatial variation and directionality are now expressible at the mesoscale, so only the microscale parameters remain isotropic and uniform per node.

## Waviness as a third surface scale

**Decision.** A surface carries `waviness` and `wavinessLength` alongside the microscale statistics. Their ratio defines the mesoscale gradient, which reduces the bearing region before resolving the finish. Waviness must be the longer and gentler scale.

**Separate scale.** σ and *ℓ* describe load-bearing asperities and their traversal rate. Waviness limits the region where nominally flat faces approach closely enough to contact. A cast face may have 1e-4 m roughness over 1e-3 m and 2e-4 m mould-form variation over 2e-2 m, giving the latter a gradient smaller by an order of magnitude.

**Why it matters audibly.** Without it, a large flat face bears over its whole polygon, which puts so many asperities under load that their fluctuations average away and the contact renders silent. The area a load actually bears over is smaller than the polygon by orders of magnitude, and the count of load-bearing asperities follows it. Applying the area law once at each scale, which is Persson's magnification in two discrete steps, recovers a bearing population small enough to fluctuate.

**Why the ordering is required rather than recommended.** Applying the same area reduction twice at one scale is not a degraded result but a meaningless one: the second application is describing the finish a second time under another name. Making it a MUST lets an implementation trust the two scales are separated instead of testing for it.

**Why absent means flat.** A surface with no waviness bears across the whole shared polygon, which is the idealized geometry contact-mechanics simulations use and the limit the two-step reduction collapses to. That keeps the field optional without leaving its absence undefined.

**Revisit when.** Add anisotropic or spatially varying waviness together with the corresponding generalization of microscale parameters. `normalTexture` already provides spatial variation at the mesoscale.

## Mesoscale field in the element container

**Decision.** Represent the mesoscale as a field band across the full element ribbon. Each element samples one lift, so crest elements bear load while the face-wide envelope spans valleys. A relief map derives this band from its heights; waviness parameters supply it when no map exists. The relief track separately drives contact datum motion because a wide contact produces no such motion from the periodic field.

**Resolved field.** The statistical route reduces bearing area through the mesoscale gradient before resolving the finish. The element container instead resolves the mesoscale field directly. Applying both methods to the same band would count its area reduction twice. The field determines bearing from local crest positions across a footprint.

**Relief-map band.** A relief map provides a realization of the scale described statistically by waviness. Its field controls bearing area and variation across a footprint; a scalar track cannot. Derive the band from map heights because the map provides no separate band parameters.

**Relief-map limitation.** The relief track follows a fixed path across the map, so its heights represent the map without contact-position alignment. The across-ribbon direction is synthesized from the trace spectrum as for a measured profile. Future sampling in the contact frame could make lift position-dependent.

**Datum measurement.** A contact spanning the periodic field reaches every element, so its support envelope remains at the field maximum during traversal. On a 12 cm face with a 10 cm field, removing track-driven datum motion reduces the mesoscale channel by **50 dB**. The track therefore supplies datum motion and the field supplies bearing variation. With the track retained, three relief scenes change by **+0.5, +0.2, and −0.7 dB**, measuring bearing variation alone.

**Revisit when.** A footprint smaller than the field sweep can receive duplicate datum motion from the track and field. A future position-dependent lift applied during spring evaluation could replace the datum channel, with the existing DC length defining a possible band split.

## Mechanism-based organization

**Decision.** Sustained contact is specified as *roughness excitation*, one mechanism, with scraping and rolling as names for regions of a continuous parameter space. Impact excitation and friction-induced vibration are the other two mechanisms, one defined elsewhere in the spec and one excluded.

**Why.** The computer-graphics-audio literature frames this phenomenologically, as scraping and rolling, and taking those categories structurally causes real failures. Ren et al. classify contacts by whether tangential velocity is nonzero, which routes pure rolling away from their surface model entirely. The same habit collapses per-body sweep into one speed, which silences a box sliding on a floor: the box's own sweep is zero because the same material region stays in contact, and only the floor's is nonzero.

Two engineering fields converge on a mechanism framing, independently of each other and of the graphics literature. Rolling-noise prediction for railways and tyres has treated surface roughness as the excitation since Remington 1987. Friction acoustics calls the same phenomenon *roughness noise*, arising from asperity impacts in light contact, and separates it from the instability-driven noise of strongly loaded contact (Akay 2002). That second split is our scope boundary, arrived at by a field that had no interest in our problem.

**Contact filter.** Rolling-noise literature uses this term for attenuation of roughness wavelengths shorter than the contact patch. It reports a cutoff that decreases with speed and stronger filtering for softer contacts, consistent with deriving patch radius from effective modulus.

**And it independently supports treating the excitation as a vector.** Sliding surfaces are observed to develop contact forces with components in both tangential and normal directions, each driving its own response, and the partitioning between them varies with normal load (Akay 2002). A scalar force along the contact normal cannot express that.

## Rolling from surface traversal

**Decision.** No rolling-specific force term. Rolling is the same contact with zero slip, and the patch filter runs on every contact rather than being switched on for rolling.

**Why.** Attribution matters here because the pieces come from different places. van den Doel et al. 2001 identify the mechanism, attributing rolling's low-frequency character to collisions just in front of a very small contact area, and relate collision velocity to the contact region, but leave the cutoff as a free parameter to tune. Deriving it from the Hertz patch radius closes that parameter using constants the contact stiffness already needs, and drops the assumption that the rolling body is a sphere.

Agarwal et al. 2021 supply the force algebra and a rolling term from the offset between a ball's center of mass and its geometric center. That term is omitted: it presumes a sphere. They also note that the balance between scraping and rolling contributions "likely depends on how much slip is present" and then leave it to hand-set free parameters. Their motions are prescribed analytically rather than simulated, so slip is not a quantity their model produces. A rigid-body solver computes slip and sweep exactly, which turns their free parameter into a derived quantity. That is the strongest argument for reporting two speeds rather than one.

**A one-speed classifier fails here.** Ren et al. classify a contact as lasting when relative tangential velocity is nonzero and transient otherwise, then apply their surface representation only to lasting contacts. Pure rolling has zero tangential velocity, so their own criterion routes it away from the surface model and into an impulse sequence generated at tessellated-geometry contacts, which makes rolling sound vary with mesh density. Slip alone cannot separate resting from rolling.

**Deliberately not duplicated.** Restitution and friction stay in `KHR_physics_rigid_bodies`. Unlike mass, where acoustic and gameplay values legitimately differ, restitution has no acoustic reading distinct from its mechanical one, so a second copy would only create conflicts.

**Naming hazard documented.** A render material's `roughness` is dimensionless optical microfacet statistics tuned by eye. This one is a physical length measured with a profilometer, orders of magnitude coarser. Neither derives from the other, and the spec says so where an author will look.

## Traversal is indexed by distance along the path

**Decision.** A surface's track is read at the distance the contact has travelled along that surface. A contact that retraces its path reads fresh surface rather than the heights it came from.

**Why not signed displacement.** Signed displacement is right about the physics and unreachable in the data model. A track is a height field, so retracing should re-read it, and the excursion of a rocking body should sound like a rattle rather than a hiss. But indexing by a *signed* quantity needs a direction to be signed against, and a one-dimensional track has none. Every candidate fails on a path a rigid-body scene can produce: the instantaneous sweep direction is never negative relative to itself, a fixed axis is silent for motion across it, a smoothed reference introduces a time constant that has to sit between the reversal and curvature timescales, and a separable pair of readers makes the excitation 3 dB louder diagonally than axially. A requirement written in terms of "traversal direction" names something a one-dimensional track cannot define.

**Distance-indexing properties.** Distance has no directional dependence in amplitude or correlation and requires no parameter. Asperity encounter rate scales with speed irrespective of sign, preserving bandwidth and level. Retracing does not reproduce the prior fine structure because a track represents a path rather than a surface.

**Requirements for removing the limitation.** The two surface scales require different representations.

The **microscale finish** has no two-dimensional data. Agarwal et al. 2021 use a stored two-dimensional field over the contact excursion. Covering unbounded sliding costs (*T* *f*<sub>Nyquist</sub>)² samples for *T* seconds of non-repeating output, around 6 × 10⁸ for one second at 48 kHz. Extent and resolution scale inversely with speed, so the count is speed-independent. A one-dimensional track needs only *T* *f*<sub>Nyquist</sub> samples. Procedural evaluation stores no field and synthesizes only the octaves resolved by current speed and load.

The **mesoscale relief** is already a field: `normalTexture` is a two-dimensional texture and the contact state already carries **p** to sample it at. Its costs are bounded and known rather than open. Measured over the Khronos sample assets, a single-channel height field with a full mip pyramid is 11 MB for a 2048² map and under 1 MB for more than half of them. A screened Poisson solve at derivation time gives the height from the normal map path-independently, which is the two-dimensional form of the leak this spec already prescribes and removes the drift that motivates it, leaving the tangential term to read the map's slope directly with no integration at all. Reading it at **p** rather than along a path is the better model, and the reason this spec does not yet require it is that obtaining a texture coordinate at the contact needs the contacting geometry to be the textured mesh, which a simplified collider is not.

## Tangential force split by mechanism

**Decision.** Use two tangential excitation terms. The geometric term projects contact load onto local surface tilt along each surface's sweep direction. The frictional term applies Coulomb traction to the force fluctuation along the slip direction. Neither term has a free parameter.

Crossing a tilted asperity converts normal load into tangential force along surface traversal. Coulomb traction acts along slip and is bounded by *μN*. Separate terms preserve these directions.

**Their v is the contact's velocity across the surface, not the slip.** Agarwal et al. 2021 write *f*<sub>h</sub> = β₁|**v** · ∇*S*|^β₂. Reading that **v** as the slip makes the term vanish for pure rolling, which their own rolling equation contradicts by keeping it. That one substitution is the difference between rolling sounding and rolling falling silent.

**Velocity-dependent component.** With Hunt and Crossley damping, *f*<sub>n</sub>∂*h* expands to *k*δ^(3/2)∂*h* + *k*δ^(3/2)*c*<sub>d</sub>δ̇∂*h*. The second term is proportional to the height-change rate used by Agarwal et al. Matching the expressions identifies β₁ with *f*<sub>n</sub>*c*<sub>d</sub>∂*h*, with units N·s·m⁻¹. A fitted β₁ combines load, dissipation, and surface slope. Projecting the full contact force provides the parameter-free form.

**Consequence for the contact state.** The force model acts along directions rather than magnitudes, so the state carries slip and sweep as velocities, and the speeds the requirements are written in terms of are their magnitudes.

**The geometric term reverses per surface, which reads as a sign convention and is a physical result.** Take the one arrangement where the interface is unambiguous: a smooth box sliding on a bumpy floor. The box has no sweep of its own, since the same material region stays in contact, so only the floor's surface is traversed, and the contact normal is simply the floor's. The box climbing a rising asperity is retarded and the floor is driven forward. Swapping which body carries the asperity, a bumpy ball rolling on a smooth floor, is the same mechanics with the roles exchanged and so demands the opposite sign for the same slope. A body is therefore driven along **û**ᵢ by its own surface's slope and against it by the other body's, and the two bodies receive equal and opposite excitations as Newton's third law requires. Summing the two surfaces with one sign, which is the form that first suggests itself from *h* = *h*₁ + *h*₂, satisfies neither arrangement.

**Boundary combination.** Averaging the surfaces gives slope (∂*h*₁ − ∂*h*₂)/2 and halves the magnitude. A smooth body follows the full rough-surface profile. Superpose the two single-rough-surface cases to retain full magnitude and antisymmetry.

**The audible reach is narrow, and the correctness is not.** Within one body the two surfaces' slopes are independent, so a difference and a sum have the same root-mean-square and an uncorrelated pair sounds the same either way. It is when both bodies of a contact sound that a common sign adds two excitations that should cancel.

**Coordinate frames.** Speeds are frame-independent, but vector subtraction requires a common frame. Kinematic relations therefore use a common frame before each body's state is transformed locally. Position uses the full inverse transform; directions and velocities use inverse rotation only. Preserving velocity magnitude is required because `sampleSpacing` is absolute.

## Vibrational coupling is MAY

**Decision.** Coupling the contact force to the body's own modal displacement is optional.

**Why.** Coupling produces micro-collisions, chattering, and contact-dependent damping (Zheng and James 2011), which an open-loop force cannot reproduce. It requires each force sample to depend on the previous summed modal state, serializing an otherwise per-mode-parallel update. The established FoleyAutomatic open-loop model remains a valid fallback, so coupling is optional.

**Linear superposition remains valid.** The force generator contains the nonlinearity, while the downstream mode bank remains linear. The existing requirement to superpose excitations applies to the bank.

**Bilateral coupling.** A full treatment shares one separation between two sounding bodies. The specification also permits each body to couple only to its own modal state.

## Scope exclusions

Everything excluded *composes with* the modal core rather than replacing it:

- **Radiation transfer** — exactly separable: per-mode scalar amplitude field multiplying oscillator output (uncontested from PAT through KleinPAT). WaveBlender proves the same modal payload drives a full wave solve with no extra data. Future layer: per-mode FFAT-style cube maps (~2 KB/mode as 8-bit textures) map naturally onto glTF's image machinery.
- **Propagation/spatialization/listeners** — `KHR_audio_graph`/`KHR_audio_emitter`/platform territory; this extension only produces a monophonic source at the node.
- **Contact-event plumbing** — matches KHR_physics_rigid_bodies' own scoping.
- **Friction-induced vibration** (squeal, bowing, brake noise) — velocity-weakening friction driving a limit cycle, which needs its own bristle state and is a different mechanism from riding over asperities. Composes as an additional force term on the same contact.
- **Anisotropic and spatially varying microscale finish, and surface wear** — the statistical parameters are isotropic and uniform per node, with variation and directionality expressible one scale up through `normalTexture`. Wear would make a surface dynamic, which conflicts with static document data.
- **Thin-shell mode coupling, fracture, sample hybrids** — known limitations of linear modal synthesis, documented as validity bounds (Authoring Notes) rather than half-specified features.
- **Analysis itself** — authoring-time; appendix documents the standard pipeline informatively.

**Declarative data with fixed semantics.** The specification provides precomputed results and defines renderer behavior. Every adopted system in the survey uses this structure; MPEG-4 Structured Audio instead standardized a programmable synthesis language and saw no adoption.

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
- Add a `KHR_interactivity` excitation node for impact and sustained-contact state after that ecosystem stabilizes.
- MeshEditor conformance: retain eigenvector 3-vectors in `mesh2modes`, import/export of the extension, direction-aware excitation, sustained-contact rendering.
