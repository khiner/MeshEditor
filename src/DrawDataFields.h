#pragma once

#include <array>
#include <cstdint>
#include <string_view>

enum class FieldType : uint8_t {
    U32,
};

struct DrawDataField {
    FieldType Type;
    std::string_view Name;
    std::string_view DefaultValue;
};

constexpr std::array DrawDataFields{
    DrawDataField{FieldType::U32, "VertexSlot", "InvalidSlot"},
    DrawDataField{FieldType::U32, "IndexSlot", "InvalidSlot"},
    DrawDataField{FieldType::U32, "IndexOffset", "0"},
    DrawDataField{FieldType::U32, "ModelSlot", "InvalidSlot"},
    DrawDataField{FieldType::U32, "FirstInstance", "0"},
    DrawDataField{FieldType::U32, "ObjectIdSlot", "InvalidSlot"},
    DrawDataField{FieldType::U32, "FaceNormalSlot", "InvalidSlot"},
    DrawDataField{FieldType::U32, "FaceIdOffset", "0"},
    DrawDataField{FieldType::U32, "FaceNormalOffset", "0"},
    DrawDataField{FieldType::U32, "VertexCountOrHeadImageSlot", "0"},
    DrawDataField{FieldType::U32, "ElementIdOffset", "0"},
    DrawDataField{FieldType::U32, "ElementStateSlot", "InvalidSlot"},
    DrawDataField{FieldType::U32, "VertexOffset", "0"},
    DrawDataField{FieldType::U32, "Pad0", "0"},
    DrawDataField{FieldType::U32, "Pad1", "0"},
    DrawDataField{FieldType::U32, "Pad2", "0"},
};
