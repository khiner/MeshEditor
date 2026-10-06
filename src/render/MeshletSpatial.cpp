#include "render/MeshletSpatial.h"
#include "Parallel.h"
#include "Profile.h"

#include "mesh/Mesh.h"
#include "mesh/MeshStore.h"
#include "numeric/dvec3.h"
#include "state/Scene.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {
using Node = MeshletSpatialNode;
using numeric::dvec3;

uvec2 VolumeBits(double value) {
    const auto bits = std::bit_cast<uint64_t>(value);
    return {uint32_t(bits), uint32_t(bits >> 32u)};
}
double UnpackVolume(uvec2 words) { return std::bit_cast<double>(uint64_t(words.y) << 32u | words.x); }

struct VolumeSource {
    std::span<const Vertex> Vertices;
    TriangleVertexView Triangles;
};
VolumeSource Source(const state::Scene &r, const MeshStore::Record &owner) {
    const auto &meshes = r.Context.get<const MeshStore>();
    return {meshes.Arenas().Vertices.Buffer.GetSpan<Vertex>(), Mesh{meshes, owner.StoreId}.TriangleVertices()};
}
double MeshletVolume(const RenderArenas &gpu, const VolumeSource &source, uint32_t id) {
    const auto meshlet = gpu.Meshlets.Get({id, 1u})[0];
    double sum = 0.0;
    for (const auto triangle : gpu.MeshletTriangleIds.Get({meshlet.TriangleOffset, meshlet.TriangleCount})) {
        const auto corners = source.Triangles.TriangleAtHandle(triangle);
        const dvec3 a{source.Vertices[corners[0]].Position}, b{source.Vertices[corners[1]].Position}, c{source.Vertices[corners[2]].Position};
        sum += Dot(a, Cross(b, c)) / 6.0;
    }
    return sum;
}

AABB Join(AABB a, AABB b) { return {Min(a.Min, b.Min), Max(a.Max, b.Max)}; }
AABB Bounds(const MeshletRecord &meshlet) {
    const vec3 radius{meshlet.Radius};
    return {meshlet.Center - radius, meshlet.Center + radius};
}
uint32_t Priority(uint32_t id) {
    uint32_t x = id + 0x9e3779b9u;
    x = (x ^ (x >> 16u)) * 0x85ebca6bu;
    x = (x ^ (x >> 13u)) * 0xc2b2ae35u;
    return x ^ (x >> 16u);
}
bool Higher(uint32_t a, uint32_t b) {
    const auto pa = Priority(a), pb = Priority(b);
    return pa == pb ? a < b : pa < pb;
}
uint64_t Key(vec3 center) {
    if (!std::isfinite(center.x) || !std::isfinite(center.y) || !std::isfinite(center.z)) throw std::invalid_argument("Spatial meshlet center must be finite.");
    const auto ordered = [](float value) {
        const auto bits = std::bit_cast<uint32_t>(value);
        return (bits & 0x80000000u ? ~bits : bits ^ 0x80000000u) >> 11u;
    };
    const auto x = ordered(center.x), y = ordered(center.y), z = ordered(center.z);
    uint64_t key = 0u;
    for (uint32_t bit = 0u; bit < 21u; ++bit)
        key |= uint64_t((x >> bit) & 1u) << (bit * 3u) |
            uint64_t((y >> bit) & 1u) << (bit * 3u + 1u) |
            uint64_t((z >> bit) & 1u) << (bit * 3u + 2u);
    return key;
}
uint64_t Key(const Node &node) { return uint64_t(node.Key.y) << 32u | node.Key.x; }
uvec2 KeyWords(uint64_t key) { return {uint32_t(key), uint32_t(key >> 32u)}; }
bool Before(uint64_t a, uint32_t ai, uint64_t b, uint32_t bi) { return a == b ? ai < bi : a < b; }

// The root is a local copy, which the caller records only when the edit moved it.
struct Tree {
    RenderArenas &Gpu;
    uint32_t Root;
    VolumeSource Source;
    std::unordered_map<uint32_t, Node> Pending;
    Node Read(uint32_t id) const {
        if (const auto it = Pending.find(id); it != Pending.end()) return it->second;
        return Gpu.MeshletSpatialNodes.Get({id, 1u})[0];
    }
    void Write(uint32_t id, Node node) { Pending.insert_or_assign(id, node); }
    void Flush() {
        if (Pending.empty()) return;
        auto &buffer = Gpu.MeshletSpatialNodes.Buffer;
        std::vector<uint32_t> ids;
        ids.reserve(Pending.size());
        for (const auto &[id, node] : Pending) ids.push_back(id);
        buffer.CaptureWriteElements(ids, sizeof(Node));
        buffer.GetMutableSpan<Node>({ids.front(), 1u}); // Wait for the batched page copy before changing live bytes.
        auto *nodes = reinterpret_cast<Node *>(buffer.Contents().data());
        for (const auto &[id, node] : Pending) nodes[id] = node;
    }
    void Fix(uint32_t id) {
        auto node = Read(id);
        auto box = Bounds(Gpu.Meshlets.Get({id, 1u})[0]);
        double volume = UnpackVolume(node.LocalVolume);
        if (node.Left != InvalidOffset) {
            const auto child = Read(node.Left);
            box = Join(box, child.Box);
            volume += UnpackVolume(child.SubtreeVolume);
        }
        if (node.Right != InvalidOffset) {
            const auto child = Read(node.Right);
            box = Join(box, child.Box);
            volume += UnpackVolume(child.SubtreeVolume);
        }
        node.Box = box;
        node.SubtreeVolume = VolumeBits(volume);
        Write(id, node);
    }
    void FixUp(uint32_t id) {
        while (id != InvalidOffset) {
            Fix(id);
            id = Read(id).Parent;
        }
    }
    void Link(uint32_t parent, uint32_t old_child, uint32_t child) {
        if (parent == InvalidOffset) {
            Root = child;
            return;
        }
        auto node = Read(parent);
        if (node.Left == old_child) node.Left = child;
        else if (node.Right == old_child) node.Right = child;
        else throw std::logic_error("Spatial parent does not own its child.");
        Write(parent, node);
    }
    void RotateLeft(uint32_t id) {
        auto a = Read(id), b = Read(a.Right);
        const auto right = a.Right, parent = a.Parent;
        a.Right = b.Left;
        a.Parent = right;
        if (a.Right != InvalidOffset) {
            auto child = Read(a.Right);
            child.Parent = id;
            Write(a.Right, child);
        }
        b.Left = id;
        b.Parent = parent;
        Write(id, a);
        Write(right, b);
        Link(parent, id, right);
        Fix(id);
        Fix(right);
    }
    void RotateRight(uint32_t id) {
        auto a = Read(id), b = Read(a.Left);
        const auto left = a.Left, parent = a.Parent;
        a.Left = b.Right;
        a.Parent = left;
        if (a.Left != InvalidOffset) {
            auto child = Read(a.Left);
            child.Parent = id;
            Write(a.Left, child);
        }
        b.Right = id;
        b.Parent = parent;
        Write(id, a);
        Write(left, b);
        Link(parent, id, left);
        Fix(id);
        Fix(left);
    }
    void Insert(uint32_t id) {
        const auto record = Gpu.Meshlets.Get({id, 1u})[0];
        if (record.RefinedGroup != InvalidOffset || record.Topology != 0u) throw std::invalid_argument("Spatial index only accepts finest triangle meshlets.");
        const auto key = Key(record.Center);
        const auto volume = VolumeBits(MeshletVolume(Gpu, Source, id));
        Write(id, {.Box = Bounds(record), .Key = KeyWords(key), .Meshlet = id, .LocalVolume = volume, .SubtreeVolume = volume});
        if (Root == InvalidOffset) {
            Root = id;
            return;
        }
        auto parent = Root;
        for (;;) {
            auto value = Read(parent);
            const bool left = Before(key, id, Key(value), parent);
            auto &child = left ? value.Left : value.Right;
            if (child != InvalidOffset) {
                parent = child;
                continue;
            }
            child = id;
            Write(parent, value);
            auto inserted = Read(id);
            inserted.Parent = parent;
            Write(id, inserted);
            break;
        }
        while (parent != InvalidOffset && Higher(id, parent)) {
            if (Read(parent).Left == id) RotateRight(parent);
            else RotateLeft(parent);
            parent = Read(id).Parent;
        }
        FixUp(id);
    }
    void Remove(uint32_t id) {
        if (id >= Gpu.MeshletSpatialNodes.Buffer.Count<Node>() || Read(id).Meshlet != id) throw std::logic_error("Spatial meshlet is not indexed.");
        for (;;) {
            const auto node = Read(id);
            if (node.Left == InvalidOffset && node.Right == InvalidOffset) break;
            if (node.Left == InvalidOffset ||
                (node.Right != InvalidOffset && Higher(node.Right, node.Left))) RotateLeft(id);
            else RotateRight(id);
        }
        const auto parent = Read(id).Parent;
        Link(parent, id, InvalidOffset);
        Write(id, Node{});
        FixUp(parent);
    }
};

} // namespace

void BuildMeshletSpatial(state::Scene &r, std::span<MeshStore::Record *const> owners) {
    auto &gpu = r.Context.get<MeshStore>().Render();
    std::vector<std::vector<uint32_t>> finest(owners.size());
    for (uint32_t i = 0u; i < owners.size(); ++i) {
        const auto &owner = *owners[i];
        if (owner.SpatialRoot != InvalidOffset) throw std::logic_error("Spatial tree already exists.");
        if (owner.ExtrasFaces.Count) continue;
        auto &ids = finest[i];
        ids.reserve(owner.Level0Count);
        gpu.ActiveMeshlets.ForEach(owner.MeshletRoot, [&](uint32_t id) {
            const auto record = gpu.Meshlets.Get({id, 1u})[0];
            if (record.RefinedGroup == InvalidOffset && record.Topology == 0u) ids.push_back(id);
        });
        if (!ids.empty()) gpu.MeshletSpatialNodes.Buffer.CaptureWrite(uint64_t(owner.Meshlets.Offset) * sizeof(Node), uint64_t(owner.Meshlets.Count) * sizeof(Node));
    }
    // Each owner writes only its own node range, so the trees build concurrently after the serial history capture.
    auto *all = reinterpret_cast<Node *>(gpu.MeshletSpatialNodes.Buffer.Contents().data());
    ParallelFor(uint32_t(owners.size()), [&](uint32_t i) {
        auto &ids = finest[i];
        if (ids.empty()) return;
        auto &owner = *owners[i];
        const auto source = Source(r, owner);
        const std::span nodes{all + owner.Meshlets.Offset, owner.Meshlets.Count};
        std::ranges::fill(nodes, Node{});
        for (const auto id : ids) {
            const auto record = gpu.Meshlets.Get({id, 1u})[0];
            const auto volume = VolumeBits(MeshletVolume(gpu, source, id));
            nodes[id - owner.Meshlets.Offset] = {.Box = Bounds(record), .Key = KeyWords(Key(record.Center)), .Meshlet = id, .LocalVolume = volume, .SubtreeVolume = volume};
        }
        const auto read = [&](uint32_t id) -> Node & { return nodes[id - owner.Meshlets.Offset]; };
        std::ranges::sort(ids, [&](uint32_t a, uint32_t b) { return Before(Key(read(a)), a, Key(read(b)), b); });
        std::vector<uint32_t> stack;
        stack.reserve(ids.size());
        for (const auto id : ids) {
            uint32_t last = InvalidOffset;
            while (!stack.empty() && Higher(id, stack.back())) {
                last = stack.back();
                stack.pop_back();
            }
            if (!stack.empty()) {
                read(stack.back()).Right = id;
                read(id).Parent = stack.back();
            }
            if (last != InvalidOffset) {
                read(id).Left = last;
                read(last).Parent = id;
            }
            stack.push_back(id);
        }
        owner.SpatialRoot = stack.front();
        std::vector<uint32_t> order{owner.SpatialRoot}, postorder;
        postorder.reserve(ids.size());
        while (!order.empty()) {
            const auto id = order.back();
            order.pop_back();
            postorder.push_back(id);
            const auto &node = read(id);
            if (node.Left != InvalidOffset) order.push_back(node.Left);
            if (node.Right != InvalidOffset) order.push_back(node.Right);
        }
        for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
            auto &node = read(*it);
            auto box = Bounds(gpu.Meshlets.Get({*it, 1u})[0]);
            double volume = UnpackVolume(node.LocalVolume);
            if (node.Left != InvalidOffset) {
                const auto &child = read(node.Left);
                box = Join(box, child.Box);
                volume += UnpackVolume(child.SubtreeVolume);
            }
            if (node.Right != InvalidOffset) {
                const auto &child = read(node.Right);
                box = Join(box, child.Box);
                volume += UnpackVolume(child.SubtreeVolume);
            }
            node.Box = box;
            node.SubtreeVolume = VolumeBits(volume);
        }
    });
}

namespace {
// Runs `edit` over the owner's tree and records the root only when the edit moved it.
void EditTree(state::Scene &r, const MeshStore::Record &owner, auto &&edit) {
    auto &meshes = r.Context.get<MeshStore>();
    Tree tree{meshes.Render(), owner.SpatialRoot, Source(r, owner)};
    edit(tree);
    tree.Flush();
    if (tree.Root != owner.SpatialRoot) meshes.WriteRecord(owner.StoreId).SpatialRoot = tree.Root;
}
} // namespace

void ReplaceMeshletSpatial(state::Scene &r, const MeshStore::Record &owner, std::span<const uint32_t> removed, std::span<const uint32_t> added) {
    const profile::CpuScope scope{"ReplaceMeshletSpatial"};
    EditTree(r, owner, [&](Tree &tree) {
        for (const auto id : removed) tree.Remove(id);
        for (const auto id : added) tree.Insert(id);
    });
}

void RefitMeshletSpatial(state::Scene &r, const MeshStore::Record &owner, std::span<const uint32_t> changed) {
    EditTree(r, owner, [&](Tree &tree) {
        for (const auto id : changed) {
            tree.Remove(id);
            tree.Insert(id);
        }
    });
}

SpatialSurfacePoint ClosestMeshletPoint(const RenderArenas &gpu, const MeshStore::Record &owner, std::span<const Vertex> vertices, TriangleVertexView triangles, vec3 point) {
    SpatialSurfacePoint best;
    if (owner.SpatialRoot == InvalidOffset) return best;
    float best_distance2 = std::numeric_limits<float>::max();
    uint32_t best_triangle = InvalidOffset;
    const auto distance2 = [point](AABB box) {
        const vec3 outside = Max(Max(box.Min - point, point - box.Max), vec3{0});
        return Dot(outside, outside);
    };
    const auto child_distance = [&](uint32_t id) {
        return id == InvalidOffset ? std::numeric_limits<float>::max() :
                                     distance2(gpu.MeshletSpatialNodes.Get({id, 1u})[0].Box);
    };
    uint32_t current = owner.SpatialRoot, previous = InvalidOffset;
    while (current != InvalidOffset) {
        const auto node = gpu.MeshletSpatialNodes.Get({current, 1u})[0];
        if (previous == node.Parent && distance2(node.Box) > best_distance2) {
            previous = current;
            current = node.Parent;
            continue;
        }
        const bool left_first = child_distance(node.Left) <= child_distance(node.Right);
        const auto near = left_first ? node.Left : node.Right;
        const auto far = left_first ? node.Right : node.Left;
        auto next = node.Parent;
        if (previous == node.Parent) {
            const auto meshlet = gpu.Meshlets.Get({current, 1u})[0];
            for (const auto triangle : gpu.MeshletTriangleIds.Get({meshlet.TriangleOffset, meshlet.TriangleCount})) {
                const auto ids = triangles.TriangleAtHandle(triangle);
                const auto hit = ClosestPointOnTriangle(point, vertices[ids[0]].Position, vertices[ids[1]].Position, vertices[ids[2]].Position);
                const vec3 delta = hit.Position - point;
                const float value = Dot(delta, delta);
                if (value < best_distance2 || (value == best_distance2 && triangle < best_triangle)) {
                    best_distance2 = value;
                    best_triangle = triangle;
                    best = {ids, hit.Weights};
                }
            }
            next = near != InvalidOffset ? near : far;
        } else if (previous == near) next = far;
        previous = current;
        current = next;
    }
    return best;
}

std::optional<double> MeshletEnclosedVolume(const RenderArenas &gpu, const MeshStore::Record &owner, const Mesh &mesh) {
    if (owner.SpatialRoot == InvalidOffset || !mesh.HasClosedSurface()) return std::nullopt;
    return std::abs(UnpackVolume(gpu.MeshletSpatialNodes.Get({owner.SpatialRoot, 1u})[0].SubtreeVolume));
}
