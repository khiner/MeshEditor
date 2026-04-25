#pragma once

#include "Camera.h"
#include "mesh/MeshData.h"
#include "numeric/vec3.h"

#include <cstdint>
#include <vector>

struct PunctualLight;

constexpr uint8_t VClassNone = 0, VClassBillboard = 1, VClassSpotCone = 2, VClassScreenspace = 3, VClassGroundPoint = 4;

struct ExtrasWireframe {
    MeshData Data;
    std::vector<uint8_t> VertexClasses{}; // Empty == all VCLASS_NONE.

    uint32_t AddVertex(vec3 pos, uint8_t vclass);
    void AddEdge(uint32_t a, uint32_t b);
    void AddCircle(float radius, uint32_t segments, float z, uint8_t vclass, uint32_t edge_stride = 1);
    void AddDiamond(float radius, uint8_t vclass, vec3 axis1, vec3 axis2, vec3 center = {});
};

MeshData BuildCameraFrustumMesh(const Camera &);
ExtrasWireframe BuildLightMesh(const PunctualLight &);
