#pragma once

#include "mesh/halfedge/Handle.h"

#include <string>
#include <vector>

enum class MeshElement {
    None,
    Vertex, // Vertices are not duplicated. Each vertex uses the vertex normal.
    Edge, // Vertices are duplicated. Each vertex uses the vertex normal.
    Face, // Vertices are duplicated for each face. Each vertex uses the face normal.
};

inline const std::vector AllElements{MeshElement::Vertex, MeshElement::Edge, MeshElement::Face};

constexpr std::string to_string(MeshElement element) {
    switch (element) {
        case MeshElement::Vertex: return "vertex";
        case MeshElement::Edge: return "edge";
        case MeshElement::Face: return "face";
        case MeshElement::None: return "none";
    }
}

struct MeshElementIndex {
    MeshElementIndex(MeshElement element = MeshElement::None, uint index = he::null) : Element(element), Index(index) {}

    MeshElement Element;
    uint Index;

    uint operator*() const { return Index; }
    bool operator==(const MeshElementIndex &) const = default;
    operator bool() const { return Index != he::null; }
};

struct MeshElementIndexHash {
    size_t operator()(const MeshElementIndex &mei) const {
        return std::hash<uint>{}(static_cast<uint>(mei.Element)) ^ (std::hash<uint>{}(mei.Index) << 1);
    }
};
