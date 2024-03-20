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

inline std::string to_string(MeshElement element) {
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

    auto operator<=>(const MeshElementIndex &) const = default;

    bool is_valid() const { return Index >= 0; }
    int idx() const { return Index; }
};
