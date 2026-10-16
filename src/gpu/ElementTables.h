#pragma once

#include "gpu/SlotOffset.h"

// A record's table run in each element domain's table arena, through which its record-relative handles resolve to arena addresses.
struct ElementTables {
    SlotOffset Vertices DEFAULT(), Halfedges DEFAULT(), Edges DEFAULT(), Faces DEFAULT(), Triangles DEFAULT();
};
