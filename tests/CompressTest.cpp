#include "Compress.h"

#include <boost/ut.hpp>

#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace boost::ut;

namespace fs = std::filesystem;

namespace {
void Write(const fs::path &path, std::string_view bytes) {
    fs::create_directories(path.parent_path());
    std::ofstream out{path, std::ios::binary};
    out.write(bytes.data(), std::streamsize(bytes.size()));
}

std::vector<std::byte> ReadAll(const fs::path &path) {
    std::ifstream in{path, std::ios::binary | std::ios::ate};
    std::vector<std::byte> bytes(size_t(in.tellg()));
    in.seekg(0);
    in.read(reinterpret_cast<char *>(bytes.data()), std::streamsize(bytes.size()));
    return bytes;
}

// Every regular file under `dir`, keyed by its relative path.
std::map<std::string, std::vector<std::byte>> Snapshot(const fs::path &dir) {
    std::map<std::string, std::vector<std::byte>> files;
    for (const auto &e : fs::recursive_directory_iterator{dir}) {
        if (e.is_regular_file()) files[fs::relative(e.path(), dir).generic_string()] = ReadAll(e.path());
    }
    return files;
}
} // namespace

int main() {
    "compress/decompress round trip"_test = [] {
        const auto root = fs::temp_directory_path() / "MeshEditor-compress-test";
        fs::remove_all(root);
        const auto src = root / "src", archive = root / "out.project", dst = root / "dst";

        Write(src / "a.txt", "hello");
        Write(src / "nested/deep/b.bin", std::string{'\0', '\x01', '\x7f', '\0', 'z', '\0'}); // embedded nulls
        Write(src / "empty", "");
        Write(src / "big", std::string(3'000'000, 'x')); // spans multiple stream chunks

        expect(Compress(src, archive));
        expect(Decompress(archive, dst));
        const bool round_trips = Snapshot(src) == Snapshot(dst);
        expect(round_trips);

        fs::remove_all(root);
    };

    "compress fails on a missing directory"_test = [] {
        const auto missing = fs::temp_directory_path() / "MeshEditor-compress-missing";
        fs::remove_all(missing);
        expect(!Compress(missing, missing / "out.project"));
    };
}
