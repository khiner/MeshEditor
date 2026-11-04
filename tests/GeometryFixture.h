#pragma once

#include "Paths.h"
#include "mesh/GeometryRefresh.h"
#include "mesh/MeshCreate.h"
#include "mesh/MeshTopology.h"
#include "state/Registry.h"

#include <boost/ut.hpp>
#include <print>

// A registry with a mesh store on a Metal context and no editor, for headless geometry operations.
struct GeometryFixture {
    NS::SharedPtr<NS::AutoreleasePool> Pool{NS::TransferPtr(NS::AutoreleasePool::alloc()->init())};
    mtl::Context Context;
    mtl::BindlessSet Slots{Context};
    mtl::BufferContext Buffers{Context, Slots};
    state::Registry Scene;
    GeometryFixture() {
        Scene.Resources.emplace<mtl::LibraryCache>(Context, Paths::Shaders());
        Scene.Resources.emplace<MeshStore>(Buffers);
    }
    MeshStore &Store() { return Scene.Resources.get<MeshStore>(); }
    Mesh Get(uint32_t id) { return Mesh{Store(), id}; }
    void Audit() {
        std::string why;
        const bool valid = Store().Audit(why);
        if (!valid) std::println("audit: {}", why);
        boost::ut::expect(valid);
    }
    uint32_t Create(MeshSource source) {
        const auto id = CreateMesh(Scene, std::move(source)).StoreId;
        ShadeNewMeshes(Scene);
        Audit();
        return id;
    }
    uint32_t Create(MeshData data) { return Create(MeshSource{.Data = std::move(data)}); }
    GeometryTopologyResult Run(MeshTopologyTask task) {
        auto result = ExecuteGeometryTopology(Scene, std::span{&task, 1u}).front();
        Audit();
        return result;
    }
};
// The quad over [-1, 1] in x and y.
inline MeshData UnitQuad() { return MeshData{{{-1.f, -1.f, 0.f}, {1.f, -1.f, 0.f}, {1.f, 1.f, 0.f}, {-1.f, 1.f, 0.f}}, {{0u, 1u, 2u, 3u}}}; }
