# Architecture & engineering policies

- Target Apple Silicon on macOS 26 or later.
    - Use host-visible unified-memory buffers for direct CPU/GPU writes.
    - Store each GPU-readable live value exactly once, in its UMA buffer.
      Capture changed pages before in-place CPU or GPU writes.
- UI and event code emits actions whose `Apply` handlers mutate registry state.
    - Direct writes are also permitted in Derived or reactive systems, engine/GPU write-back, and background-worker continuations.
- Use ordered containers in Persistent components.
- `Project::AfterRestore()` reconciles Derived state and must not mutate Persistent state.
- Use current formats for MeshEditor-owned files and records.
  Add legacy readers, migrations, compatibility encodings, or old-format tests only when explicitly requested.
- Put an unbraced `if` body on the same line as its condition.
  Use braces when the body starts on a new line.
