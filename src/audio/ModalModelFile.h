#pragma once

#include "ContactModel.h"
#include "ModalEigenSummary.h"
#include "ModalModes.h"
#include "mesh/TetMeshData.h"

#include <filesystem>
#include <optional>

// A modal solve's results. Stored in a write-once, content-hashed file, so a path's contents never
// change and replaying an action log reproduces the same bytes.
struct ModalModelData {
    ModalModes Modes;
    MassProperties Mass;
    TetMeshData Tets;
    ModalEigenSummary Summary;

    bool operator==(const ModalModelData &) const = default;
};

// The modal results store.
std::filesystem::path ModalModelsDir();

// Writes `data` to a content-addressed file under ModalModelsDir() and returns its path relative
// to it, reusing the existing file when identical content is already stored. Empty on IO failure.
std::filesystem::path SaveModalModelFile(const ModalModelData &);

std::optional<ModalModelData> LoadModalModelFile(const std::filesystem::path &relative);
