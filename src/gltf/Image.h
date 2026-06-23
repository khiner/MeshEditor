#pragma once

#include "gltf/MimeType.h"

#include <string>
#include <vector>

namespace gltf {
struct Image {
    // Encoded source bytes. Retained for embedded sources so save can passthrough byte-identically.
    // External-URI images drop Bytes after upload - SourceAbsPath is the persistence and is re-read at save time.
    std::vector<std::byte> Bytes;
    MimeType MimeType;
    std::string Name, Uri{};
    std::string SourceAbsPath{}; // resolved absolute path, only set when Uri is non-empty.
    bool SourceDataUri{}, SourceHadMimeType{};
    // Flip true when the GPU texture for this image is mutated.
    // SaveGltf then re-encodes from the GPU pixels instead of using Bytes / re-reading the source file.
    bool IsDirty{};
};
} // namespace gltf