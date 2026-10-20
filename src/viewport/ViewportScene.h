#pragma once

#include "state/Entity.h"
#include "viewport/ViewportSchema.h"

// The scene a viewport entity shows and its actions act on, beside the viewport's Name.
struct ViewportScene {
    state::Entity Scene;
};
