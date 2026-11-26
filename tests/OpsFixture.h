#pragma once

#include "GeometryFixture.h"
#include "MeshWalk.h"
#include "ops/Evaluate.h"

#include <random>

inline constexpr float Frame{2.f}, Time{0.5f};
// The mesh domains in the order a declaration tries them for its first compiling domain.
inline constexpr std::array MeshDomains{ops::Domain::Vertex, ops::Domain::Corner, ops::Domain::Face, ops::Domain::Edge};

// Whether the type has a Metal type, so a program holds its values.
inline bool HasValue(mx::Type type) { return !mx::MetalType(type).empty(); }
inline bool HasValues(const mx::Declaration &d) {
    return std::ranges::all_of(d.Outputs, [](const mx::Port &p) { return HasValue(p.Type); });
}

// A random float from a spread with zero of both signs, small integers and halves, values across zero and one, and large magnitudes.
inline uint32_t RandomFloatWord(std::mt19937 &rng) {
    constexpr std::array Special{0.f, -0.f, 1.f, -1.f, 0.5f, -0.5f, 2.f, -2.f, 0.25f, 3.f};
    const auto kind = rng() % 8u;
    if (kind == 0u) return std::bit_cast<uint32_t>(Special[rng() % Special.size()]);
    if (kind == 1u) {
        const auto magnitude = std::pow(10.f, std::uniform_real_distribution{1.f, 6.f}(rng));
        return std::bit_cast<uint32_t>(rng() % 2u ? magnitude : -magnitude);
    }
    return std::bit_cast<uint32_t>(std::uniform_real_distribution{-2.5f, 2.5f}(rng));
}
// Random words of a value of the type, with every small integer and both booleans.
inline std::vector<uint32_t> RandomWords(mx::Type type, std::mt19937 &rng) {
    std::vector<uint32_t> words(mx::Words(type));
    for (auto &word : words) {
        word = type == mx::Type::Boolean ? rng() % 2u :
            type == mx::Type::Integer    ? std::bit_cast<uint32_t>(int32_t(rng() % 13u) - 4) :
                                           RandomFloatWord(rng);
    }
    return words;
}

// A mesh with the op drivers, their program registry and a value store.
struct OpsFixture : GeometryFixture {
    explicit OpsFixture(MeshSource source = {.Data = UnitQuad()}, std::optional<MTL::MathMode> math_mode = {})
        : Id{Create(std::move(source))}, Programs{Scene.Resources.emplace<mtl::ProgramRegistry>(Context, math_mode)} {}

    uint32_t Id;
    mtl::ProgramRegistry &Programs;
    ops::Drivers &Drivers{ops::GetDrivers(Scene)};
    BufferArena<uint32_t> Values{Buffers, SlotType::Buffer};
    uint32_t Attributes{};

    uint32_t Index(const ops::Program &program) { return mtl::AddPrograms(Programs, std::array{program.Source}).front(); }
    SlotOffset Block(std::span<const uint32_t> words) { return {Values.Buffer.Slot, Values.Allocate(words).Offset}; }
    uint32_t AddAttribute(ops::Domain domain, AttributeType type) { return Store().AddAttribute(Id, ops::AttributeDomainOf(domain), type, std::format("a{}", Attributes++)).value(); }
    // The record's handles in the domain, which a new record holds densely.
    uint32_t Count(ops::Domain domain) {
        const auto mesh = Get(Id);
        return domain == ops::Domain::Vertex ? mesh.VertexCount() : domain == ops::Domain::Edge ? mesh.EdgeCount() :
            domain == ops::Domain::Face                                                        ? mesh.FaceCount() :
                                                                                                 mesh.HalfEdgeCount();
    }
    // Records the program at every element of the domain into new attributes, and returns them.
    std::vector<uint32_t> RecordElements(mtl::ComputeChain &chain, ops::Domain domain, const ops::Program &program, uint32_t index, SlotOffset values, std::span<const OpImageTexture> images = {}) {
        std::vector<uint32_t> attributes;
        std::vector<NamedAttributeRef> outputs;
        for (const auto type : program.Outputs) {
            attributes.push_back(AddAttribute(domain, *ops::StorageOf(type)));
            outputs.push_back(Store().AttributeRef(attributes.back()));
        }
        // Each attribute the program reads is the record's attribute of its name, storage type and domain.
        std::vector<uint32_t> read;
        for (const auto &attribute : program.Attributes) {
            const auto found = Store().FindAttribute(Id, attribute.Name);
            const bool held = found != InvalidOffset && Store().GetAttribute(found).Domain == ops::AttributeDomainOf(domain) && Store().GetAttribute(found).Type == ops::StorageOf(attribute.Type);
            read.push_back(held ? found : InvalidOffset);
        }
        ops::RecordElements(chain, Drivers, Store(), Id, domain, {program, index}, values, ops::AttributeTable(Store(), domain, read), images, outputs, Frame, Time);
        return attributes;
    }
    // The words of the attributes at each element of the domain, in attribute order.
    std::vector<std::vector<uint32_t>> ElementWords(std::span<const uint32_t> attributes, ops::Domain domain) {
        std::vector<std::vector<uint32_t>> elements(Count(domain));
        for (uint32_t handle = 0; handle < elements.size(); ++handle) {
            for (const auto attribute : attributes) {
                const auto value = test::AttributeWords(Store(), attribute, handle);
                elements[handle].append_range(std::span{value}.first(AttributeComponents(Store().GetAttribute(attribute).Type)));
            }
        }
        return elements;
    }
};
