#include "SceneViewUBO.glsl"
#include "ViewportTheme.glsl"

const uint InteractionModeObject = 0u, InteractionModeEdit = 1u, InteractionModeExcite = 2u;
const uint EditElementNone = 0u, EditElementVertex = 1u, EditElementEdge = 2u, EditElementFace = 4u;
