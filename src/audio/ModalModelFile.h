#pragma once

#include "ContactModel.h"
#include "ModalEigenSummary.h"
#include "ModalModes.h"
#include "mesh/TetMeshData.h"

#include <expected>
#include <filesystem>
#include <string>

namespace project {
struct Assets;
}

// Stores modal solve results in write-once, content-addressed files for deterministic replay.
struct ModalModelData {
    ModalModes Modes;
    MassProperties Mass;
    TetMeshData Tets;
    ModalEigenSummary Summary;

    bool operator==(const ModalModelData &) const = default;
};

// Return an immutable project asset reference.
std::expected<std::filesystem::path, std::string> SaveModalModelFile(project::Assets &, const ModalModelData &);
std::expected<ModalModelData, std::string> LoadModalModelFile(const std::filesystem::path &);
