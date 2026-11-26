#pragma once

#include <expected>
#include <vector>

namespace xml {
struct Attribute {
    std::string Name, Value;
};

struct Element {
    std::string Name;
    std::vector<Attribute> Attributes;
    std::vector<Element> Children;
};

// Parses a document into its root element, skipping the declaration, comments and text.
// Attribute values decode the five standard entities, and a '&' that starts none of them is literal.
std::expected<Element, std::string> Parse(std::string_view text);

// The document holding the root element, after an XML declaration, with each element on its own line.
// Attribute values encode the characters the parser decodes, apart from an apostrophe.
std::string Write(const Element &root);

// The value of the named attribute, empty when the element has none.
std::string_view Find(const Element &, std::string_view name);
} // namespace xml
