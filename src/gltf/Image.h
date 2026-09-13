#pragma once

#include "gltf/MimeType.h"

#include <string>
#include <vector>

namespace gltf {
struct Image {
    // Retain embedded source bytes for export when SourcePath is empty.
    std::vector<std::byte> Bytes;
    MimeType MimeType;
    std::string Name, Uri{};
    std::string SourcePath{};
    bool SourceDataUri{}, SourceHadMimeType{};
    // Selects GPU readback and re-encoding during SaveGltf.
    bool IsDirty{};
};
} // namespace gltf
