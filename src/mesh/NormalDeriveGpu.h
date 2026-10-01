#pragma once

#include "gpu/ElementWork.h"
#include "gpu/NormalDeriveEntry.h"
#include "gpu/NormalDerivePushConstants.h"
#include "metal/BufferArena.h"
#include "state/Entity.h"

struct MeshStore;
namespace mtl { struct ComputeChain; }

// Captures the canonical base normal pages the entry's work writes.
// These are its vertex and face normals and the sector normals around its vertices.
void CaptureNormalWrites(state::Scene &, const NormalDeriveEntry &, const BufferArena<uint32_t> &vertex_work, const BufferArena<uint32_t> &face_work);

// Canonical geometry and connectivity.
// Callers choose base or posed destinations.
std::optional<NormalDeriveEntry> MakeDeriveEntryInputs(const MeshStore &, uint32_t store_id);

// Records the same two-phase kernel as frame and sparse edit derivation.
// Only job metadata is uploaded, and geometry and normal values stay in their GPU arenas.
void EncodeDeriveNormals(state::Scene &, mtl::ComputeChain &, std::span<const NormalDeriveEntry>, NormalDerivePushConstants);
void DeriveNormalsNow(state::Scene &, std::span<const NormalDeriveEntry>, NormalDerivePushConstants);
void DeriveMeshNormalsNow(state::Scene &, std::span<const uint32_t> store_ids);
// Records base normals after a local change.
// Faces cover the changed face normals, and vertices cover every affected normal fan.
// Both inputs are finished canonical membership in `work`.
void EncodeDeriveMeshNormals(state::Scene &, mtl::ComputeChain &, uint32_t store_id, const BufferArena<uint32_t> &work,
                             ElementWork vertices, uint32_t vertex_count, ElementWork faces, uint32_t face_count);
