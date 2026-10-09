#pragma once

#include "gpu/Element.h"
#include "gpu/MeshElementBlock.h"
#include "gpu/SlotOffset.h"
#include <array>
#include <stdexcept>
#include <variant>
#include <vector>

// Explicit, sorted canonical handles. Empty domains select nothing.
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

// One domain's selection as resident mask blocks, where block First + b of Arena holds canonical block b.
// The arena holds a block for every block the mesh owns while an operation reads or writes it, and its masks are complete when the operation reads them.
// Bits of elements the mesh does not own select nothing.
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

struct MeshStore;
void ValidateGeometrySelection(const MeshStore &, uint32_t id, const GeometrySelection &);
// The mask words of canonical handle h start at word Offset + h / 32 of the binding.
SlotOffset MaskBinding(SelectionMasks);
// The live handles of a mesh domain its masks select, ascending.
std::vector<uint32_t> ReadMasks(const MeshStore &, uint32_t id, Element, SelectionMasks);
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
