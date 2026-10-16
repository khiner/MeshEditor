#pragma once

#include "gpu/Element.h"
#include "gpu/MeshElementBlock.h"
#include "gpu/SlotOffset.h"
#include <variant>
#include <vector>

// Explicit, sorted canonical handles.
// Empty domains select nothing.
struct GeometrySelection {
    std::vector<uint32_t> Vertices, Edges, Faces;
    const std::vector<uint32_t> &Get(Element element) const {
        switch (element) {
            case Element::Vertex: return Vertices;
            case Element::Edge: return Edges;
            case Element::Face: return Faces;
            default: throw std::invalid_argument("Geometry selection requires an element domain.");
        }
    }
};

// One bit per element of a 256-element canonical block.
using SelectionBlock = std::array<uint32_t, MeshElementBlockWords>;
template<typename T> struct BufferArena;

// Domain selection masks map Arena block First + b to canonical block b.
// Allocate masks for every owned mesh block before use and initialize them before reading.
// Bits outside the mesh's owned elements select nothing.
struct SelectionMasks {
    const BufferArena<SelectionBlock> *Arena;
    uint32_t First{};
};
struct GeometryMasks {
    SelectionMasks Vertices, Edges, Faces;
    SelectionMasks Get(Element element) const {
        switch (element) {
            case Element::Vertex: return Vertices;
            case Element::Edge: return Edges;
            case Element::Face: return Faces;
            default: throw std::invalid_argument("Geometry masks require an element domain.");
        }
    }
};
// Handle lists or resident masks.
using SelectionSource = std::variant<GeometrySelection, GeometryMasks>;

struct Mesh;
struct MeshStore;
void ValidateGeometrySelection(const MeshStore &, uint32_t id, const GeometrySelection &);
// Every live element of the mesh in each domain.
GeometrySelection AllHandles(const Mesh &);
// The mask words of canonical handle h start at word Offset + h / 32 of the binding.
SlotOffset MaskBinding(SelectionMasks);
// The live handles of a mesh domain its masks select, ascending.
std::vector<uint32_t> ReadMasks(const MeshStore &, uint32_t id, Element, SelectionMasks);
// The live handles of each mesh domain its masks select, ascending.
GeometrySelection ReadMasks(const MeshStore &, uint32_t id, const GeometryMasks &);
// The handles a source selects: its own lists, or the masked live handles read on construction.
class SelectedHandles {
public:
    SelectedHandles(const MeshStore &, uint32_t id, const SelectionSource &);
    const GeometrySelection &operator*() const { return Lists ? *Lists : Read; }
    const GeometrySelection *operator->() const { return &**this; }

private:
    const GeometrySelection *Lists;
    GeometrySelection Read;
};
