# Architecture & engineering policies

- The target is Apple Silicon on macOS 26 or later, and nothing else.
    - Unified memory is assumed: buffers are host-visible and written in place, with no staging copies.
    - Data the GPU reads lives in exactly one UMA buffer. Never copy it into a CPU-side container. Snapshots serialize the mesh arenas wholesale, so a UMA buffer persists as readily as a vector does.
    - Metal is the graphics API. Shaders are authored in MSL and there is no cross-API translation step.
    - Shaders reach textures and buffers through the bindless argument buffer, which an encoder cannot see, so Metal tracks no hazard for them. A command buffer's passes are ordered by `mtl::PassChain`.
- User actions never mutate registry state outside of an action's `Apply` handler — UI/event code emits actions.
    - Direct writes are only for Apply, derived/reactive systems, engine/GPU write-back, background-worker continuations.
- A Persistent component must not contain an unordered container.

- Position edits preserve canonical vertices during preview and evaluate final action parameters at commit, including in headless execution.
    - Changed vertices expand through existing face/fan adjacency into affected normal outputs, level-zero meshlets, and 256-vertex bounds tiles. One UMA mask/active-word representation supplies GPU dispatches and CPU consumers.
    - Posed storage and bounds partials initialize when their layout changes, then update incrementally across gestures. Edited meshes stay at the finest LOD on every instance until the existing full builder runs when leaving edit mode.
    - Position-change events reach BVH refit, curvature, physics, and audio before trackers clear. Topology and shading-layout changes invalidate edit work. Volume, hulls, and other global consumers retain their required full calculations.
    - Snapshots serialize directly from arena spans while preserving the owning snapshot format.
