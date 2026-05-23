#pragma once

#include "gpu/Element.h"
#include "gpu/InteractionMode.h"

struct Interaction {
    InteractionMode Mode{InteractionMode::Object};
};

struct EditMode {
    Element Value{Element::Vertex};
};
