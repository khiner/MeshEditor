# Architecture & engineering policies

- Target Apple Silicon on macOS 26 or later.
    - Use host-visible unified-memory buffers for direct CPU/GPU writes.
    - Store each GPU-readable live value exactly once, in its UMA buffer.
      Capture changed pages before in-place CPU or GPU writes.
- UI and event code emits actions whose `Apply` handlers mutate document state.
    - Direct writes are also permitted in Derived or reactive systems, engine/GPU write-back, and background-worker continuations.
- `state::Registry` owns entity allocation, native component pages, domain resources, and dirty sets.
    - Component and resource access uses fixed compile-time schema slots, with no runtime type lookup.
    - Keep component definitions, registration, and encoding in their owning domain TUs.
    - Each component is Persistent, Workspace, or Derived, declared in its domain's snapshot registration.
      Workspace components belong to the Persistent component that defines their entity's kind, are emplaced and removed with it, and belong to the identity through its returns from history.
      A fresh identity starts from defaults.
    - Reads return const values.
      Use `edit` to capture before a write, or `patch` to also publish an update.
      Mutable views capture each value they expose; use const views for reading.
    - Create/update/destroy publish lifecycle changes.
      Destroy handlers observe the old value.
      A scene reset drops components without handlers, and each resource holding scene state declares `Reset(T &, Registry &)` to restore its post-init state.
      Managed dirty sets discard destroyed identities; explicit removal trackers retain them until consumed.
    - History owns native copies of changed CPU records and shares unchanged trie branches.
      Serialization is for hashes and persistent records, not hot value installation.
    - Entity generation/free-list state is versioned once.
      Exhausted generations retire their slot.
      Navigation and reset advance a separate epoch used to reject stale worker results.
- Use ordered containers in Persistent components.
- `Project::AfterRestore()` reconciles Derived state and must not mutate Persistent state.
- After any committed action, a cold restore and a replay of the history, each with the live workspace applied, reproduce the live application exactly, including the composed UI pixels.
- MeshEditor-owned files and records use one unversioned current format.
  Keep no schema revision fields, schema compatibility checks, legacy readers, migrations, or old-format tests.
  Imported formats follow their external specifications.
