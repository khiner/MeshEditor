#include "materialx/Xml.h"

#include <algorithm>
#include <format>

namespace xml {
namespace {
struct Reader {
    std::string_view Text;
    size_t Pos{0};

    bool AtEnd() const { return Pos >= Text.size(); }
    bool Next(std::string_view s) const { return Text.substr(Pos).starts_with(s); }
    void SkipSpace() {
        while (!AtEnd() && std::string_view{" \t\r\n"}.contains(Text[Pos])) ++Pos;
    }
    std::unexpected<std::string> Error(std::string_view message) const {
        const auto line = 1 + std::ranges::count(Text.substr(0, Pos), '\n');
        return std::unexpected{std::format("line {}: {}", line, message)};
    }
    // Moves past `close`, returning false when the text ends first.
    bool SkipPast(std::string_view close) {
        const auto end = Text.find(close, Pos);
        if (end == std::string_view::npos) return false;
        Pos = end + close.size();
        return true;
    }
    std::string_view Name() {
        const auto start = Pos;
        while (!AtEnd() && !std::string_view{" \t\r\n/>=\"'"}.contains(Text[Pos])) ++Pos;
        return Text.substr(start, Pos - start);
    }
};

constexpr std::array<std::pair<std::string_view, char>, 5> Entities{{{"&lt;", '<'}, {"&gt;", '>'}, {"&amp;", '&'}, {"&quot;", '"'}, {"&apos;", '\''}}};

std::string Decode(std::string_view raw) {
    std::string out;
    out.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        const auto entity = raw[i] == '&' ? std::ranges::find_if(Entities, [&](const auto &e) { return raw.substr(i).starts_with(e.first); }) : Entities.end();
        if (entity == Entities.end()) {
            out.push_back(raw[i]);
        } else {
            out.push_back(entity->second);
            i += entity->first.size() - 1;
        }
    }
    return out;
}

std::string Encode(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (const char c : text) {
        // An apostrophe stays literal inside the double quotes that delimit values.
        const auto entity = c == '\'' ? Entities.end() : std::ranges::find(Entities, c, &std::pair<std::string_view, char>::second);
        if (entity == Entities.end()) out.push_back(c);
        else out += entity->first;
    }
    return out;
}

void WriteElement(const Element &element, uint32_t depth, std::string &out) {
    out.append(2u * depth, ' ');
    out += std::format("<{}", element.Name);
    for (const auto &[name, value] : element.Attributes) out += std::format(" {}=\"{}\"", name, Encode(value));
    if (element.Children.empty()) {
        out += " />\n";
        return;
    }
    out += ">\n";
    for (const auto &child : element.Children) WriteElement(child, depth + 1u, out);
    out.append(2u * depth, ' ');
    out += std::format("</{}>\n", element.Name);
}

// Skips the declaration, comments and text before the next element or closing tag.
std::expected<void, std::string> SkipMisc(Reader &r) {
    while (true) {
        while (!r.AtEnd() && r.Text[r.Pos] != '<') ++r.Pos;
        if (r.Next("<!--")) {
            if (!r.SkipPast("-->")) return r.Error("unterminated comment");
        } else if (r.Next("<?") || r.Next("<!")) {
            if (!r.SkipPast(">")) return r.Error("unterminated declaration");
        } else {
            return {};
        }
    }
}

std::expected<Element, std::string> ParseElement(Reader &r) {
    ++r.Pos; // '<'
    Element element{.Name = std::string{r.Name()}};
    if (element.Name.empty()) return r.Error("expected an element name");
    while (true) {
        r.SkipSpace();
        if (r.AtEnd()) return r.Error(std::format("unterminated tag <{}>", element.Name));
        if (r.Next("/>")) {
            r.Pos += 2;
            return element;
        }
        if (r.Next(">")) {
            ++r.Pos;
            break;
        }
        const auto name = r.Name();
        r.SkipSpace();
        if (name.empty() || !r.Next("=")) return r.Error(std::format("malformed attribute in <{}>", element.Name));
        ++r.Pos;
        r.SkipSpace();
        if (r.AtEnd() || (r.Text[r.Pos] != '"' && r.Text[r.Pos] != '\'')) return r.Error(std::format("unquoted attribute '{}'", name));
        const auto quote = r.Text[r.Pos++];
        const auto end = r.Text.find(quote, r.Pos);
        if (end == std::string_view::npos) return r.Error(std::format("unterminated attribute '{}'", name));
        element.Attributes.push_back({std::string{name}, Decode(r.Text.substr(r.Pos, end - r.Pos))});
        r.Pos = end + 1;
    }
    while (true) {
        if (auto skipped = SkipMisc(r); !skipped) return std::unexpected{std::move(skipped.error())};
        if (r.AtEnd()) return r.Error(std::format("unclosed element <{}>", element.Name));
        if (r.Next("</")) {
            r.Pos += 2;
            if (r.Name() != element.Name) return r.Error(std::format("mismatched closing tag for <{}>", element.Name));
            r.SkipSpace();
            if (!r.Next(">")) return r.Error(std::format("malformed closing tag for <{}>", element.Name));
            ++r.Pos;
            return element;
        }
        auto child = ParseElement(r);
        if (!child) return child;
        element.Children.push_back(std::move(*child));
    }
}
} // namespace

std::expected<Element, std::string> Parse(std::string_view text) {
    Reader r{text};
    if (auto skipped = SkipMisc(r); !skipped) return std::unexpected{std::move(skipped.error())};
    if (r.AtEnd()) return r.Error("no root element");
    return ParseElement(r);
}

std::string Write(const Element &root) {
    std::string out{"<?xml version=\"1.0\"?>\n"};
    WriteElement(root, 0u, out);
    return out;
}

std::string_view Find(const Element &element, std::string_view name) {
    const auto it = std::ranges::find(element.Attributes, name, &Attribute::Name);
    return it == element.Attributes.end() ? std::string_view{} : std::string_view{it->Value};
}
} // namespace xml
