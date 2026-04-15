#pragma once

#include "gpu/Element.h"
#include "gpu/InteractionMode.h"

struct SceneInteraction {
    InteractionMode Mode{InteractionMode::Object};
};

struct SceneEditMode {
    Element Value{Element::Vertex};
};
