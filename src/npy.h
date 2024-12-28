#pragma once

/*
Copied and heavily modified the bits needed for reading .npy files from
[libnpy](https://github.com/llohse/libnpy/blob/master/include/npy.hpp) (MIT License).
*/

#include <algorithm>
#include <array>
#include <complex>
#include <filesystem>
#include <format>
#include <fstream>
#include <numeric>
#include <ranges>
#include <typeindex>
#include <unordered_map>
#include <vector>

namespace npy {
/*
Compile-time test for byte order.
If your compiler does not define these per default, you may want to define one of these constants manually.
Defaults to little endian order.
*/
#if defined(__BYTE_ORDER) && __BYTE_ORDER == __BIG_ENDIAN || defined(__BIG_ENDIAN__) || defined(__ARMEB__) || \
    defined(__THUMBEB__) || defined(__AARCH64EB__) || defined(_MIBSEB) || defined(__MIBSEB) || defined(__MIBSEB__)
constexpr bool big_endian = true;
#else
constexpr bool big_endian = false;
#endif

using ndarray_len_t = uint64_t;
using shape_t = std::vector<ndarray_len_t>;
using version_t = std::pair<char, char>;

template<typename Scalar> struct npy_data {
    shape_t shape = {};
    bool fortran_order = false;
    std::vector<Scalar> data = {};
};

constexpr size_t magic_string_length = 6;
constexpr std::array<char, magic_string_length> magic_string{'\x93', 'N', 'U', 'M', 'P', 'Y'};

inline version_t read_magic(std::istream &istream) {
    std::array<char, magic_string_length + 2> buf{};
    istream.read(buf.data(), sizeof(buf));
    if (!istream) throw std::runtime_error("io error: failed reading file");
    if (!std::equal(magic_string.begin(), magic_string.end(), buf.begin())) throw std::runtime_error("this file does not have a valid npy format.");

    return {buf[magic_string_length], buf[magic_string_length + 1]};
}

inline std::string read_header(std::istream &istream) {
    // Check magic bytes an version number
    version_t version = read_magic(istream);
    uint32_t header_length = 0;
    if (version == version_t{1, 0}) {
        std::array<uint8_t, 2> header_len_le16{};
        istream.read(reinterpret_cast<char *>(header_len_le16.data()), 2);
        header_length = (header_len_le16[0] << 0) | (header_len_le16[1] << 8);

        if ((magic_string_length + 2 + 2 + header_length) % 16 != 0) {
            // TODO(llohse): display warning
        }
    } else if (version == version_t{2, 0}) {
        std::array<uint8_t, 4> header_len_le32{};
        istream.read(reinterpret_cast<char *>(header_len_le32.data()), 4);
        header_length = (header_len_le32[0] << 0) | (header_len_le32[1] << 8) | (header_len_le32[2] << 16) | (header_len_le32[3] << 24);

        if ((magic_string_length + 2 + 4 + header_length) % 16 != 0) {
            // TODO(llohse): display warning
        }
    } else {
        throw std::runtime_error("unsupported file format version");
    }

    auto buf_v = std::vector<char>(header_length);
    istream.read(buf_v.data(), header_length);
    return {buf_v.data(), header_length};
}

// Remove leading and trailing whitespace
constexpr std::string_view trim(std::string_view str) {
    constexpr std::string_view whitespace = " \t";

    auto begin = str.find_first_not_of(whitespace);
    if (begin == std::string_view::npos) return "";

    auto end = str.find_last_not_of(whitespace);
    return str.substr(begin, end - begin + 1);
}

constexpr std::string_view get_value_from_map(std::string_view mapstr) {
    size_t sep_pos = mapstr.find_first_of(":");
    if (sep_pos == std::string::npos) return "";

    return trim(mapstr.substr(sep_pos + 1));
}

// Parse the string representation of a Python dict.
// The keys need to be known and may not appear anywhere else in the data.
std::unordered_map<std::string_view, std::string_view> parse_dict(std::string_view in, const std::vector<std::string_view> &keys) {
    if (keys.size() == 0) return {};

    // Unwrap dictionary
    in = trim(in);
    if (in.front() != '{' || in.back() != '}') throw std::runtime_error("Not a Python dictionary.");

    in = in.substr(1, in.length() - 2);

    std::vector<std::pair<size_t, std::string_view>> positions;
    for (auto value : keys) {
        const size_t pos = in.find(std::format("'{}'", value));
        if (pos == std::string::npos) throw std::runtime_error(std::format("Missing '{}' key.", value));

        positions.emplace_back(pos, std::move(value));
    }

    std::ranges::sort(positions);
    std::unordered_map<std::string_view, std::string_view> map;
    for (size_t i = 0; i < positions.size(); ++i) {
        size_t begin{positions[i].first}, end{std::string_view::npos};
        auto key = positions[i].second;
        if (i + 1 < positions.size()) end = positions[i + 1].first;

        auto raw_value = trim(in.substr(begin, end - begin));
        map.emplace(key, get_value_from_map(raw_value.back() == ',' ? raw_value.substr(0, raw_value.length() - 1) : raw_value));
    }

    return map;
}

// Parse the string representation of a Python boolean
constexpr bool parse_bool(std::string_view in) {
    if (in == "True") return true;
    if (in == "False") return false;

    throw std::runtime_error("Invalid python boolan.");
}

// Parse the string representation of a Python str
constexpr std::string_view parse_str(std::string_view in) {
    if (in.front() != '\'' || in.back() != '\'') throw std::runtime_error("Invalid python string.");

    return in.substr(1, in.length() - 2);
}

// Parse the string represenatation of a Python tuple into a vector of its items
inline std::vector<std::string> parse_tuple(std::string_view in) {
    in = trim(in);
    if (in.front() != '(' || in.back() != ')') throw std::runtime_error("Invalid Python tuple.");

    std::istringstream iss(std::string{in.substr(1, in.length() - 2)});
    std::vector<std::string> v;
    for (std::string token; std::getline(iss, token, ',');) v.emplace_back(token);
    return v;
}

constexpr char little_endian_char = '<', big_endian_char = '>', no_endian_char = '|';
constexpr std::array<char, 3> endian_chars{little_endian_char, big_endian_char, no_endian_char};
constexpr std::array<char, 4> numtype_chars{'f', 'i', 'u', 'c'};

struct dtype_t {
    char byteorder, kind;
    uint32_t itemsize;

    auto operator<=>(const dtype_t &) const = default;
};

constexpr dtype_t parse_descr(std::string_view typestring) {
    if (typestring.length() < 3) throw std::runtime_error("Invalid typestring (length)");

    const char byteorder_c = typestring.at(0);
    const char kind_c = typestring.at(1);
    const auto itemsize_s = typestring.substr(2);
    if (!std::ranges::contains(endian_chars, byteorder_c)) throw std::runtime_error("Invalid typestring (byteorder)");
    if (!std::ranges::contains(numtype_chars, kind_c)) throw std::runtime_error("Invalid typestring (kind)");
    if (!std::all_of(itemsize_s.begin(), itemsize_s.end(), ::isdigit)) throw std::runtime_error("Invalid typestring (itemsize)");

    return {byteorder_c, kind_c, uint32_t(std::stoul(std::string{itemsize_s}))};
}

/*
The first 6 bytes are a magic string: exactly "x93NUMPY".
The next 1 byte is an unsigned byte: the major version number of the file format, e.g. x01.
The next 1 byte is an unsigned byte: the minor version number of the file format, e.g. x00.
Note: the version of the file format is not tied to the version of the numpy package.
The next 2 bytes form a little-endian unsigned short int: the length of the header data HEADER_LEN.
The next HEADER_LEN bytes form the header data describing the array's format.
It is an ASCII string which contains a Python literal expression of a dictionary.
It is terminated by a newline and padded with spaces ('x20') to make the total length of the
magic string + 4 + HEADER_LEN be evenly divisible by 16 for alignment purposes.
The dictionary contains three keys:
    "descr" : dtype.descr
        An object that can be passed as an argument to the numpy.dtype() constructor to create the array's dtype.
    "fortran_order" : bool
        Whether the array data is Fortran-contiguous or not.
        Since Fortran-contiguous arrays are a common form of non-C-contiguity, we allow them to be written directly to disk for efficiency.
    "shape" : tuple of int
        The shape of the array.
        For repeatability and readability, this dictionary is formatted using pprint.pformat() so the keys are in alphabetic order.
 */
struct header_t {
    dtype_t dtype;
    shape_t shape;
    bool fortran_order;
};

constexpr header_t parse_header(std::string_view header) {
    static const std::vector<std::string_view> keys{"descr", "fortran_order", "shape"};
    auto dict_map = parse_dict(header.back() == '\n' ? header.substr(0, header.length() - 1) : header, keys);
    if (dict_map.size() == 0) throw std::runtime_error("Invalid dictionary in header");

    shape_t shape;
    for (auto item : parse_tuple(dict_map["shape"])) shape.emplace_back(ndarray_len_t(std::stoul(item)));

    return {parse_descr(parse_str(dict_map["descr"])), shape, parse_bool(dict_map["fortran_order"])};
}

template<typename T>
constexpr std::type_index type_idx() { return std::type_index(typeid(T)); }
template<typename T>
constexpr std::pair<std::type_index, dtype_t> dtype_entry(char endian, char kind) {
    return {type_idx<T>(), {endian, kind, sizeof(T)}};
}

constexpr char endian_char = (big_endian ? big_endian_char : little_endian_char);
const std::unordered_map<std::type_index, dtype_t> dtype_map{
    dtype_entry<char>(no_endian_char, 'i'),
    dtype_entry<signed char>(no_endian_char, 'i'),
    dtype_entry<short>(endian_char, 'i'),
    dtype_entry<int>(endian_char, 'i'),
    dtype_entry<long>(endian_char, 'i'),
    dtype_entry<long long>(endian_char, 'i'),
    dtype_entry<unsigned char>(no_endian_char, 'u'),
    dtype_entry<unsigned short>(endian_char, 'u'),
    dtype_entry<unsigned int>(endian_char, 'u'),
    dtype_entry<unsigned long>(endian_char, 'u'),
    dtype_entry<unsigned long long>(endian_char, 'u'),
    dtype_entry<float>(endian_char, 'f'),
    dtype_entry<double>(endian_char, 'f'),
    dtype_entry<long double>(endian_char, 'f'),
    dtype_entry<std::complex<float>>(endian_char, 'c'),
    dtype_entry<std::complex<double>>(endian_char, 'c'),
    dtype_entry<std::complex<long double>>(endian_char, 'c'),
};

// `in` stream position will be after the header after returning.
template<typename Scalar> inline header_t read_header(std::istream &in) {
    auto header = parse_header(read_header(in));
    auto dtype = dtype_map.at(type_idx<Scalar>());
    if (header.dtype != dtype) throw std::runtime_error("Formatting error: typestrings do not match.");

    return header;
}

// `advance` is relative to the current `in` read position.
// `in` stream position will be after the read data after returning.
template<typename Scalar> inline npy_data<Scalar> read_npy(std::istream &in, const header_t &header, size_t advance = 0, size_t size = 0) {
    const auto &shape = header.shape;
    size_t total_size = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies());
    size_t read_size = size == 0 || size > total_size - advance ? total_size - advance : size;
    npy_data<Scalar> data{shape, header.fortran_order, std::vector<Scalar>(read_size)};
    // Seek relative to the current position.
    in.seekg(std::streamoff(sizeof(Scalar) * advance), std::ios_base::cur);
    in.read(reinterpret_cast<char *>(data.data.data()), sizeof(Scalar) * read_size);
    return data;
}
template<typename Scalar> inline npy_data<Scalar> read_npy(std::istream &in, size_t advance = 0, size_t size = 0) {
    return read_npy<Scalar>(in, read_header<Scalar>(in), advance, size);
}

template<typename Scalar> inline npy_data<Scalar> read_npy(std::filesystem::path path, size_t offset = 0, size_t size = 0) {
    std::ifstream in{path, std::ifstream::binary};
    if (!in) throw std::runtime_error(std::format("IO error: failed to open file: {}", path.string()));

    return read_npy<Scalar>(in, offset, size);
}
} // namespace npy
