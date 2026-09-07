# Architecture & engineering policies

- The target is Apple Silicon on macOS 26 or later, and nothing else.
    - Unified memory is assumed: buffers are host-visible and written in place, with no staging copies.
    - Data the GPU reads lives in exactly one UMA buffer. Never copy it into a CPU-side container. Snapshots serialize non-derived mesh arenas wholesale.
- User actions never mutate registry state outside of an action's `Apply` handler — UI/event code emits actions.
    - Direct writes are only for Apply, derived/reactive systems, engine/GPU write-back, background-worker continuations.
- A Persistent component must not contain an unordered container.
- Backward compatibility for MeshEditor-owned files and records is not required. Do not add legacy readers, migrations, compatibility encodings, or tests for old formats unless explicitly requested.
- An unbraced `if` body must share a line with its condition. Use braces when the body starts on a new line.
