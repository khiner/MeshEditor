#pragma once

#include "gltf/MimeType.h"

#include <string>
#include <vector>

namespace gltf {
struct Image {
    // Where the source document kept the encoded bytes, which the export reproduces.
    enum class SourceKind : uint8_t {
        Embedded,
        DataUri,
        External,
    };
    // Encoded bytes, empty once SourcePath names a file holding them.
    std::vector<std::byte> Bytes;
    MimeType MimeType;
    SourceKind Source{SourceKind::Embedded};
    // The source document declared MimeType, so an external export repeats it.
    bool SourceHadMimeType{};
    // Selects GPU readback and re-encoding during SaveGltf.
    bool IsDirty{};
    std::string Name;
    // The external source URI as written in the document.
    std::string Uri{};
    std::string SourcePath{};
};
} // namespace gltf
