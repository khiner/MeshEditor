#include "action/LogSerialize.h"
#include "PathSerialize.h"
#include "numeric/Serialize.h"
#include <cstring>
#include <istream>
#include <ostream>

namespace action {
void SerializeAction(const Action &a, std::ostream &out) {
    static thread_local std::vector<std::byte> buffer;
    zpp::bits::out archive{buffer};
    if (zpp::bits::failure(archive(uint32_t{0}, a))) {
        out.setstate(std::ios::failbit);
        return;
    }
    const auto len = uint32_t(archive.position() - sizeof(uint32_t));
    std::memcpy(buffer.data(), &len, sizeof len);
    out.write(reinterpret_cast<const char *>(buffer.data()), std::streamsize(archive.position()));
}

bool detail::ReadAction(std::istream &in, Action &a, std::vector<std::byte> &bytes) {
    uint32_t len;
    if (!in.read(reinterpret_cast<char *>(&len), sizeof len)) return false;
    bytes.resize(len);
    if (len && !in.read(reinterpret_cast<char *>(bytes.data()), len)) return false;
    return !zpp::bits::failure(zpp::bits::in{bytes}(a));
}
} // namespace action
