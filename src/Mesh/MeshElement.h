#pragma once

#include <string>

enum class MeshElement {
    None,
    Face, // Vertices are duplicated for each face. Each vertex uses the face normal.
    Vertex, // Vertices are not duplicated. Each vertex uses the vertex normal.
    Edge, // Vertices are duplicated. Each vertex uses the vertex normal.
};

struct MeshElementIndex {
    MeshElementIndex() : Element(MeshElement::None), Index(-1) {}
    MeshElementIndex(MeshElement element, int index) : Element(element), Index(index) {}

    MeshElement Element;
    int Index;

    auto operator<=>(const MeshElementIndex &) const = default;

    bool is_valid() const { return Index >= 0; }
    int idx() const { return Index; }

    std::string ElementName() const {
        switch (Element) {
            case MeshElement::Face: return "face";
            case MeshElement::Vertex: return "vertex";
            case MeshElement::Edge: return "edge";
            case MeshElement::None: return "none";
        }
    }
};
