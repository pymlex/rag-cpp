#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "rag_db/hnsw_index.hpp"
#include "rag_db/link_storage.hpp"


namespace rag_db {

namespace {

constexpr std::uint32_t kMagic = 0x52474442u;
constexpr std::uint32_t kVersion = 1u;

struct FileHeader {
    std::uint32_t magic;
    std::uint32_t version;
    std::uint32_t dim;
    std::uint32_t m;
    std::uint32_t ef_construction;
    std::uint32_t ef_search;
    std::uint32_t cosine;
    std::uint64_t count;
    std::uint64_t entry_point;
    std::int32_t max_level;
};

}

bool HnswIndex::save(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    FileHeader header{};
    header.magic = kMagic;
    header.version = kVersion;
    header.dim = static_cast<std::uint32_t>(dim_);
    header.m = static_cast<std::uint32_t>(m_);
    header.ef_construction = static_cast<std::uint32_t>(ef_construction_);
    header.ef_search = static_cast<std::uint32_t>(ef_search_);
    header.cosine = cosine_ ? 1u : 0u;
    header.count = static_cast<std::uint64_t>(next_label_);
    header.entry_point = static_cast<std::uint64_t>(entry_point_);
    header.max_level = max_level_;
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    const auto& raw = vectors_.raw();
    out.write(reinterpret_cast<const char*>(raw.data()), static_cast<std::streamsize>(raw.size() * sizeof(float)));
    const auto& alive = vectors_.alive_flags();
    out.write(reinterpret_cast<const char*>(alive.data()), static_cast<std::streamsize>(alive.size()));
    const auto& nodes = links_.all();
    std::uint64_t node_count = nodes.size();
    out.write(reinterpret_cast<const char*>(&node_count), sizeof(node_count));
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto& node = nodes[i];
        std::int32_t level = node.level;
        out.write(reinterpret_cast<const char*>(&level), sizeof(level));
        std::uint32_t layers = static_cast<std::uint32_t>(node.by_level.size());
        out.write(reinterpret_cast<const char*>(&layers), sizeof(layers));
        for (const auto& layer : node.by_level) {
            std::uint32_t n = static_cast<std::uint32_t>(layer.size());
            out.write(reinterpret_cast<const char*>(&n), sizeof(n));
            for (Label label : layer) {
                std::uint64_t v = static_cast<std::uint64_t>(label);
                out.write(reinterpret_cast<const char*>(&v), sizeof(v));
            }
        }
    }
    return static_cast<bool>(out);
}

bool HnswIndex::load(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    FileHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (header.magic != kMagic || header.version != kVersion) {
        return false;
    }
    dim_ = header.dim;
    m_ = header.m;
    ef_construction_ = header.ef_construction;
    ef_search_ = header.ef_search;
    cosine_ = header.cosine != 0;
    next_label_ = static_cast<Label>(header.count);
    entry_point_ = static_cast<Label>(header.entry_point);
    max_level_ = header.max_level;
    vectors_ = VectorStorage(dim_, max_elements_);
    std::vector<float> raw(next_label_ * dim_);
    in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size() * sizeof(float)));
    std::vector<std::uint8_t> alive(next_label_);
    in.read(reinterpret_cast<char*>(alive.data()), static_cast<std::streamsize>(alive.size()));
    vectors_.load_raw(raw, alive);
    links_ = LinkStorage(m_);
    std::uint64_t node_count = 0;
    in.read(reinterpret_cast<char*>(&node_count), sizeof(node_count));
    auto& nodes = links_.all_mut();
    nodes.resize(static_cast<std::size_t>(node_count));
    for (std::size_t i = 0; i < node_count; ++i) {
        std::int32_t level = 0;
        in.read(reinterpret_cast<char*>(&level), sizeof(level));
        nodes[i].level = level;
        std::uint32_t layers = 0;
        in.read(reinterpret_cast<char*>(&layers), sizeof(layers));
        nodes[i].by_level.resize(layers);
        for (std::uint32_t l = 0; l < layers; ++l) {
            std::uint32_t n = 0;
            in.read(reinterpret_cast<char*>(&n), sizeof(n));
            nodes[i].by_level[l].resize(n);
            for (std::uint32_t j = 0; j < n; ++j) {
                std::uint64_t v = 0;
                in.read(reinterpret_cast<char*>(&v), sizeof(v));
                nodes[i].by_level[l][j] = static_cast<Label>(v);
            }
        }
    }
    visited_.resize(max_elements_);
    return static_cast<bool>(in);
}

}