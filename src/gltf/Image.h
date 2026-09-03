#pragma once

#include "gltf/MimeType.h"

#include <string>
#include <vector>

namespace gltf {
struct Image {
    // Embedded sources retain encoded bytes for byte-identical saves.
    // External sources retain their absolute path and reload bytes during saves.
    std::vector<std::byte> Bytes;
    MimeType MimeType;
    std::string Name, Uri{};
    std::string SourceAbsPath{}; // Set only for nonempty Uri.
    bool SourceDataUri{}, SourceHadMimeType{};
    // Selects GPU readback and re-encoding during SaveGltf.
    bool IsDirty{};
};
} // namespace gltf
