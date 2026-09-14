# Architecture & engineering policies

- Target Apple Silicon on macOS 26 or later.
    - Use host-visible unified-memory buffers for direct CPU/GPU writes.
    - Store each GPU-readable live value exactly once, in its UMA buffer.
      Capture changed pages before in-place CPU or GPU writes.
- UI and event code emits actions whose `Apply` handlers mutate document state.
    - Direct writes are also permitted in Derived or reactive systems, engine/GPU write-back, and background-worker continuations.
- `state::Scene` owns entity allocation, native component pages, domain services, and dirty sets.
    - Component and service access uses fixed compile-time schema slots, with no runtime type lookup.
    - Keep component definitions, registration, and encoding in their owning domain TUs.
    - Reads return const values. Use `edit` to capture before a write, or `patch` to also publish an update.
      Mutable views capture each value they expose; use const views for reading.
    - Create/update/destroy publish lifecycle changes. Destroy handlers observe the old value.
      Managed dirty sets discard destroyed identities; explicit removal trackers retain them until consumed.
    - History owns native copies of changed CPU records and shares unchanged trie branches.
      Serialization is for hashes and persistent records, not hot value installation.
    - Entity generation/free-list state is versioned once. Exhausted generations retire their slot.
      Navigation and reset advance a separate epoch used to reject stale worker results.
- Use ordered containers in Persistent components.
- `Project::AfterRestore()` reconciles Derived state and must not mutate Persistent state.
- Use current formats for MeshEditor-owned files and records.
  Add legacy readers, migrations, compatibility encodings, or old-format tests only when explicitly requested.
- Put an unbraced `if` body on the same line as its condition.
  Use braces when the body starts on a new line.
