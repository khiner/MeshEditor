#pragma once

#include "ContactModel.h"
#include "ModalEigenSummary.h"
#include "ModalModes.h"
#include "mesh/TetMeshData.h"

#include <filesystem>
#include <optional>

// Stores modal solve results in write-once, content-addressed files for deterministic replay.
struct ModalModelData {
    ModalModes Modes;
    MassProperties Mass;
    TetMeshData Tets;
    ModalEigenSummary Summary;

    bool operator==(const ModalModelData &) const = default;
};

// The modal results store.
std::filesystem::path ModalModelsDir();

// Writes data under dir and returns its relative content-addressed path.
// Reuses identical stored content and returns an empty path on I/O failure.
std::filesystem::path SaveModalModelFile(const std::filesystem::path &dir, const ModalModelData &);

std::optional<ModalModelData> LoadModalModelFile(const std::filesystem::path &relative);
