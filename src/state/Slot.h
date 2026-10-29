#pragma once
#include <array>
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace state {
using TypeId = uint32_t;
template<typename T> consteval std::string_view TypeName() {
    const std::string_view text = __PRETTY_FUNCTION__;
    const auto begin = text.find("T = ") + 4;
    return text.substr(begin, text.rfind(']') - begin);
}
// A type's own name, without its namespace or template arguments.
constexpr std::string_view LeafName(std::string_view name) {
    const auto base = name.substr(0, name.find('<'));
    const auto scope = base.rfind("::");
    return scope == std::string_view::npos ? base : base.substr(scope + 2);
}
template<typename T> consteval std::string_view LeafName() { return LeafName(TypeName<T>()); }

// The type's name with every anonymous namespace removed, so a private type names like a public one.
template<typename T> inline constexpr auto CanonicalChars = [] {
    constexpr std::string_view anonymous = "(anonymous namespace)::", name = TypeName<T>();
    std::array<char, name.size()> chars{};
    size_t size = 0;
    for (size_t i = 0; i < name.size();) {
        if (name.substr(i).starts_with(anonymous)) i += anonymous.size();
        else chars[size++] = name[i++];
    }
    return std::pair{chars, size};
}();
template<typename T> consteval std::string_view CanonicalName() { return {CanonicalChars<T>.first.data(), CanonicalChars<T>.second}; }

// Each domain owns a dense slot range and lists its types in its <Domain>Schema.h header.
inline constexpr TypeId ActionCount = 50; // action
inline constexpr TypeId AnimationCount = 4; // animation
inline constexpr TypeId ArmatureCount = 15; // armature
inline constexpr TypeId AudioCount = 32; // audio
inline constexpr TypeId AudioSurfaceCount = 2; // audio/surface
inline constexpr TypeId EditorCount = 2; // editor
inline constexpr TypeId GizmoCount = 1; // gizmo
inline constexpr TypeId GltfCount = 5; // gltf
inline constexpr TypeId GpuCount = 3; // gpu
inline constexpr TypeId MeshCount = 20; // mesh
inline constexpr TypeId MetalCount = 3; // metal
inline constexpr TypeId ObjectCount = 2; // object
inline constexpr TypeId PhysicsCount = 24; // physics
inline constexpr TypeId ProcedureCount = 67; // procedure
inline constexpr TypeId ProjectCount = 2; // project
inline constexpr TypeId RenderCount = 29; // render
inline constexpr TypeId SceneCount = 23; // scene and the root camera and path types
inline constexpr TypeId SelectionCount = 13; // selection
inline constexpr TypeId SnapshotCount = 1; // snapshot
inline constexpr TypeId TimelineCount = 7; // timeline
inline constexpr TypeId UiCount = 1; // ui
inline constexpr TypeId ViewportCount = 33; // viewport and the root event, panel and main types

inline constexpr TypeId ActionBase = 0;
inline constexpr TypeId AnimationBase = ActionBase + ActionCount;
inline constexpr TypeId ArmatureBase = AnimationBase + AnimationCount;
inline constexpr TypeId AudioBase = ArmatureBase + ArmatureCount;
inline constexpr TypeId AudioSurfaceBase = AudioBase + AudioCount;
inline constexpr TypeId EditorBase = AudioSurfaceBase + AudioSurfaceCount;
inline constexpr TypeId GizmoBase = EditorBase + EditorCount;
inline constexpr TypeId GltfBase = GizmoBase + GizmoCount;
inline constexpr TypeId GpuBase = GltfBase + GltfCount;
inline constexpr TypeId MeshBase = GpuBase + GpuCount;
inline constexpr TypeId MetalBase = MeshBase + MeshCount;
inline constexpr TypeId ObjectBase = MetalBase + MetalCount;
inline constexpr TypeId PhysicsBase = ObjectBase + ObjectCount;
inline constexpr TypeId ProcedureBase = PhysicsBase + PhysicsCount;
inline constexpr TypeId ProjectBase = ProcedureBase + ProcedureCount;
inline constexpr TypeId RenderBase = ProjectBase + ProjectCount;
inline constexpr TypeId SceneBase = RenderBase + RenderCount;
inline constexpr TypeId SelectionBase = SceneBase + SceneCount;
inline constexpr TypeId SnapshotBase = SelectionBase + SelectionCount;
inline constexpr TypeId TimelineBase = SnapshotBase + SnapshotCount;
inline constexpr TypeId UiBase = TimelineBase + TimelineCount;
inline constexpr TypeId ViewportBase = UiBase + UiCount;
inline constexpr TypeId SchemaSize = ViewportBase + ViewportCount;

// A domain's component and resource types in slot order.
template<TypeId Base, TypeId Count, typename... Ts> struct Schema {
    static_assert(sizeof...(Ts) == Count, "update the domain's count");
    template<typename T> static constexpr bool Has = (... || std::is_same_v<T, Ts>);
    template<typename T>
        requires Has<T>
    static consteval TypeId Of() {
        constexpr bool listed[]{std::is_same_v<T, Ts>...};
        TypeId i = 0;
        while (!listed[i]) ++i;
        return Base + i;
    }
};
template<typename T> inline constexpr TypeId Slot = SchemaSize;
template<typename T> consteval TypeId Type() {
    static_assert(Slot<std::remove_cv_t<T>> < SchemaSize, "list this type in its domain's Schema header");
    return Slot<std::remove_cv_t<T>>;
}

// Serialized records identify a type by a hash of its canonical name, so slot numbers can change without invalidating them.
enum class TypeKey : uint32_t {};
constexpr uint32_t HashName(std::string_view name) {
    uint32_t hash{2166136261u};
    for (const char c : name) hash = (hash ^ uint8_t(c)) * 16777619u;
    return hash;
}
template<typename T> consteval TypeKey Key() { return TypeKey{HashName(CanonicalName<T>())}; }

} // namespace state
