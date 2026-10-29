#pragma once

#include "action/Core.h"

#include <string>
#include <variant>

namespace action::scene {
// Creates an empty scene, unless the document holds MaxScenes.
struct Add {
    std::string Name;
};
// Removes the scene and the objects in no other scene, unless it is the only scene.
struct Remove {
    state::Entity Scene;
};
struct Rename {
    state::Entity Scene;
    std::string Name;
};
// Adds the scene to the scenes of each selected node's root.
struct LinkSelected {
    state::Entity Scene;
    Target Target{OnSelected{}};
};
// Removes the scene from the scenes of each selected node's root that is in another scene.
struct UnlinkSelected {
    state::Entity Scene;
    Target Target{OnSelected{}};
};

using Action = std::variant<Add, Remove, Rename, LinkSelected, UnlinkSelected>;

void Apply(state::Registry &, const Action &);
} // namespace action::scene
