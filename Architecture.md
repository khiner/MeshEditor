# Architecture & engineering policies

- The target is Apple Silicon on macOS 26 or later, and nothing else.
    - Unified memory is assumed: buffers are host-visible and written in place, with no staging copies.
    - Data the GPU reads lives in exactly one UMA buffer. Never copy it into a CPU-side container. Snapshots serialize the mesh arenas wholesale, so a UMA buffer persists as readily as a vector does.
    - Metal is the graphics API. Shaders are authored in MSL and there is no cross-API translation step.
    - Shaders reach textures and buffers through the bindless argument buffer, which an encoder cannot see, so Metal tracks no hazard for them. A command buffer's passes are ordered by `mtl::PassChain`.
- User actions never mutate registry state outside of an action's `Apply` handler — UI/event code emits actions.
    - Direct writes are only for Apply, derived/reactive systems, engine/GPU write-back, background-worker continuations.
- A Persistent component must not contain an unordered container.
