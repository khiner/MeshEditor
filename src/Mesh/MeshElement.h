#pragma once

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
    MeshElementIndex() : Element(MeshElement::None), Index(-1) {}
    MeshElementIndex(MeshElement element, int index) : Element(element), Index(index) {}

    MeshElement Element;
    int Index;

    bool IsValid() const { return Index >= 0; }

    int operator*() const { return Index; }
    bool operator==(const MeshElementIndex &) const = default;
};

struct MeshElementIndexHash {
    size_t operator()(const MeshElementIndex &mei) const {
        return std::hash<int>{}(static_cast<int>(mei.Element)) ^ (std::hash<int>{}(mei.Index) << 1);
    }
};
