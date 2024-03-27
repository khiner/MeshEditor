#pragma once

/*
This code is from [libnpy](https://github.com/llohse/libnpy/blob/master/include/npy.hpp) (MIT License).
I copied the bits needed for reading .npy files, with minor changes.
*/

#include <algorithm>
#include <array>
#include <complex>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
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

using ndarray_len_t = unsigned long int;
using shape_t = std::vector<ndarray_len_t>;
using version_t = std::pair<char, char>;

template<typename Scalar> struct npy_data {
    shape_t shape = {};
    bool fortran_order = false;
    std::vector<Scalar> data = {};
};

struct dtype_t {
    char byteorder, kind;
    unsigned int itemsize;

    inline std::string str() const {
        std::stringstream ss;
        ss << byteorder << kind << itemsize;
        return ss.str();
    }
    inline std::tuple<const char, const char, const unsigned int> tie() const {
        return std::tie(byteorder, kind, itemsize);
    }
};

struct header_t {
    dtype_t dtype;
    shape_t shape;
    bool fortran_order;
};

constexpr size_t magic_string_length = 6;
constexpr std::array<char, magic_string_length> magic_string = {'\x93', 'N', 'U', 'M', 'P', 'Y'};

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
    std::string header(buf_v.data(), header_length);

    return header;
}

// Remove leading and trailing whitespace
inline std::string trim(const std::string &str) {
    static const std::string whitespace = " \t";

    auto begin = str.find_first_not_of(whitespace);
    if (begin == std::string::npos) return "";

    auto end = str.find_last_not_of(whitespace);
    return str.substr(begin, end - begin + 1);
}

inline std::string get_value_from_map(const std::string &mapstr) {
    size_t sep_pos = mapstr.find_first_of(":");
    if (sep_pos == std::string::npos) return "";

    return trim(mapstr.substr(sep_pos + 1));
}

// Parse the string representation of a Python dict.
// The keys need to be known and may not appear anywhere else in the data.
inline std::unordered_map<std::string, std::string> parse_dict(std::string in, const std::vector<std::string> &keys) {
    std::unordered_map<std::string, std::string> map;
    if (keys.size() == 0) return map;

    // Unwrap dictionary
    in = trim(in);
    if (in.front() != '{' || in.back() != '}') throw std::runtime_error("Not a Python dictionary.");

    in = in.substr(1, in.length() - 2);

    std::vector<std::pair<size_t, std::string>> positions;
    for (const auto &value : keys) {
        size_t pos = in.find("'" + value + "'");
        if (pos == std::string::npos) throw std::runtime_error("Missing '" + value + "' key.");

        positions.emplace_back(pos, value);
    }

    std::sort(positions.begin(), positions.end());
    for (size_t i = 0; i < positions.size(); ++i) {
        size_t begin{positions[i].first}, end{std::string::npos};

        std::string key = positions[i].second;
        if (i + 1 < positions.size()) end = positions[i + 1].first;

        std::string raw_value = trim(in.substr(begin, end - begin));
        if (raw_value.back() == ',') raw_value.pop_back();

        map.emplace(key, get_value_from_map(raw_value));
    }

    return map;
}

// Parse the string representation of a Python boolean
inline bool parse_bool(const std::string &in) {
    if (in == "True") return true;
    if (in == "False") return false;

    throw std::runtime_error("Invalid python boolan.");
}

// Parse the string representation of a Python str
inline std::string parse_str(const std::string &in) {
    if (in.front() != '\'' || in.back() != '\'') throw std::runtime_error("Invalid python string.");

    return in.substr(1, in.length() - 2);
}

// Parse the string represenatation of a Python tuple into a vector of its items
inline std::vector<std::string> parse_tuple(std::string in) {
    static const char seperator = ',';

    in = trim(in);
    if (in.front() != '(' || in.back() != ')') throw std::runtime_error("Invalid Python tuple.");

    std::istringstream iss(in.substr(1, in.length() - 2));
    std::vector<std::string> v;
    for (std::string token; std::getline(iss, token, seperator);) v.emplace_back(token);
    return v;
}

constexpr char little_endian_char = '<', big_endian_char = '>', no_endian_char = '|';
constexpr std::array<char, 3> endian_chars = {little_endian_char, big_endian_char, no_endian_char};
constexpr std::array<char, 4> numtype_chars = {'f', 'i', 'u', 'c'};

template<typename T, size_t N> inline bool in_array(T val, const std::array<T, N> &arr) {
    return std::find(std::begin(arr), std::end(arr), val) != std::end(arr);
}

inline dtype_t parse_descr(std::string typestring) {
    if (typestring.length() < 3) throw std::runtime_error("Invalid typestring (length)");

    char byteorder_c = typestring.at(0);
    char kind_c = typestring.at(1);
    std::string itemsize_s = typestring.substr(2);
    if (!in_array(byteorder_c, endian_chars)) throw std::runtime_error("Invalid typestring (byteorder)");
    if (!in_array(kind_c, numtype_chars)) throw std::runtime_error("Invalid typestring (kind)");
    if (!std::all_of(itemsize_s.begin(), itemsize_s.end(), ::isdigit)) throw std::runtime_error("Invalid typestring (itemsize)");

    unsigned int itemsize = std::stoul(itemsize_s);
    return {byteorder_c, kind_c, itemsize};
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
inline header_t parse_header(std::string header) {
    if (header.back() == '\n') header.pop_back();

    static const std::vector<std::string> keys{"descr", "fortran_order", "shape"};
    auto dict_map = parse_dict(header, keys);
    if (dict_map.size() == 0) throw std::runtime_error("Invalid dictionary in header");

    shape_t shape;
    for (auto item : parse_tuple(dict_map["shape"])) shape.emplace_back(ndarray_len_t(std::stoul(item)));

    return {parse_descr(parse_str(dict_map["descr"])), shape, parse_bool(dict_map["fortran_order"])};
}

constexpr char host_endian_char = (big_endian ? big_endian_char : little_endian_char);
const std::unordered_map<std::type_index, dtype_t> dtype_map = {
    {std::type_index(typeid(float)), {host_endian_char, 'f', sizeof(float)}},
    {std::type_index(typeid(double)), {host_endian_char, 'f', sizeof(double)}},
    {std::type_index(typeid(long double)), {host_endian_char, 'f', sizeof(long double)}},
    {std::type_index(typeid(char)), {no_endian_char, 'i', sizeof(char)}},
    {std::type_index(typeid(signed char)), {no_endian_char, 'i', sizeof(signed char)}},
    {std::type_index(typeid(short)), {host_endian_char, 'i', sizeof(short)}},
    {std::type_index(typeid(int)), {host_endian_char, 'i', sizeof(int)}},
    {std::type_index(typeid(long)), {host_endian_char, 'i', sizeof(long)}},
    {std::type_index(typeid(long long)), {host_endian_char, 'i', sizeof(long long)}},
    {std::type_index(typeid(unsigned char)), {no_endian_char, 'u', sizeof(unsigned char)}},
    {std::type_index(typeid(unsigned short)), {host_endian_char, 'u', sizeof(unsigned short)}},
    {std::type_index(typeid(unsigned int)), {host_endian_char, 'u', sizeof(unsigned int)}},
    {std::type_index(typeid(unsigned long)), {host_endian_char, 'u', sizeof(unsigned long)}},
    {std::type_index(typeid(unsigned long long)), {host_endian_char, 'u', sizeof(unsigned long long)}},
    {std::type_index(typeid(std::complex<float>)), {host_endian_char, 'c', sizeof(std::complex<float>)}},
    {std::type_index(typeid(std::complex<double>)), {host_endian_char, 'c', sizeof(std::complex<double>)}},
    {std::type_index(typeid(std::complex<long double>)), {host_endian_char, 'c', sizeof(std::complex<long double>)}}
};

template<typename Scalar> npy_data<Scalar> inline read_npy(std::istream &in, size_t offset = 0, size_t size = 0) {
    std::string header_s = read_header(in);
    header_t header = parse_header(header_s);
    dtype_t dtype = dtype_map.at(std::type_index(typeid(Scalar)));
    if (header.dtype.tie() != dtype.tie()) throw std::runtime_error("Formatting error: typestrings do not match.");

    auto &shape = header.shape;
    size_t total_size = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies());
    size_t read_size = size == 0 || size > total_size - offset ? total_size - offset : size;
    npy_data<Scalar> data{shape, header.fortran_order, std::vector<Scalar>(read_size)};

    // Seek relative to the current position (after the header).
    in.seekg(std::streamoff(sizeof(Scalar) * offset), std::ios_base::cur);
    in.read(reinterpret_cast<char *>(data.data.data()), sizeof(Scalar) * read_size);
    return data;
}

template<typename Scalar> inline npy_data<Scalar> read_npy(const std::string &filename, size_t offset = 0, size_t size = 0) {
    std::ifstream stream(filename, std::ifstream::binary);
    if (!stream) throw std::runtime_error("IO error: failed to open a file.");

    return read_npy<Scalar>(stream, offset, size);
}
} // namespace npy
