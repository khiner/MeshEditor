#pragma once

#include "graph/Graph.h"
#include "metal/Mipmaps.h"

#include <map>
#include <memory>

namespace graph {
// The textures of an image: a texture of its texels, row by row from the top, with a mip chain, a view whose channels past the image's components read zero, and an opaque view of them for display.
struct ImageViews {
    mtl::Texture Texture, Sampled, Shown;
    mtl::MipChain Levels;
};
// The image of an occurrence of a Bake node or a filter, which programs sample through its Sampled view.
// A view that draws it shares its textures.
struct OccurrenceImage {
    std::shared_ptr<ImageViews> Views;
    uint32_t Components{};
};
// The images that evaluations at the cooks' frame keep for each mesh's occurrences of Bake nodes and filters, by mesh and occurrence.
// An entry without views is an occurrence whose last such evaluation held no image.
// They derive from the graphs and what they read, and drop when the registry's epoch advances at a restore or a reset.
struct OccurrenceImages {
    std::map<std::pair<state::Entity, NodePath>, OccurrenceImage> Images;
    uint64_t Epoch{};
};
// The registry's images of occurrences of Bake nodes and filters, after dropping those of an earlier epoch.
OccurrenceImages &CurrentImages(state::Registry &);
} // namespace graph
