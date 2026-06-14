# Architecture & engineering policies

- User actions never mutate registry state outside of an action's `Apply` handler — UI/event code emits actions.
    - Direct writes are only for Apply, derived/reactive systems, engine/GPU write-back, background-worker continuations.
