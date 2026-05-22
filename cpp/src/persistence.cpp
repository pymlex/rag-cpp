#include "ragdb/persistence.hpp"

#include <fstream>
#include <cstdint>

namespace ragdb {

namespace {

constexpr std::uint32_t kMagic = 0x52474442u;
constexpr std::uint32_t kVersion = 1;

struct FileHeader {
    std::uint32_t magic;
    std::uint32_t version;
    std::size_t dim;
    std::size_t count;
    std::int32_t M;
    std::int32_t M_max0;
    std::int32_t ef_construction;
    std::int32_t ef_search;
    Id entry_point;
    Level max_level;
};

void write_links(std::ofstream& out, const LinkStore& links, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        Level lvl = links.level(static_cast<Id>(i));
        out.write(reinterpret_cast<const char*>(&lvl), sizeof(lvl));
        for (Level layer = 0; layer <= lvl; ++layer) {
            const auto& nbrs = links.neighbors(static_cast<Id>(i), layer);
            auto n = static_cast<std::uint32_t>(nbrs.size());
            out.write(reinterpret_cast<const char*>(&n), sizeof(n));
            for (Id nid : nbrs) {
                out.write(reinterpret_cast<const char*>(&nid), sizeof(nid));
            }
        }
    }
}

void read_links(std::ifstream& in, LinkStore& links, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        Level lvl = 0;
        in.read(reinterpret_cast<char*>(&lvl), sizeof(lvl));
        links.set_level(static_cast<Id>(i), lvl);
        for (Level layer = 0; layer <= lvl; ++layer) {
            std::uint32_t n = 0;
            in.read(reinterpret_cast<char*>(&n), sizeof(n));
            auto& nbrs = links.neighbors_mut(static_cast<Id>(i), layer);
            nbrs.resize(n);
            for (std::uint32_t j = 0; j < n; ++j) {
                Id nid = 0;
                in.read(reinterpret_cast<char*>(&nid), sizeof(nid));
                nbrs[j] = nid;
            }
        }
    }
}

}

bool persist_save(
    const std::string& path,
    const HnswParams& params,
    const VectorStore& store,
    const LinkStore& links) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    FileHeader h{};
    h.magic = kMagic;
    h.version = kVersion;
    h.dim = params.dim;
    h.count = static_cast<std::size_t>(store.max_id());
    h.M = params.M;
    h.M_max0 = params.M_max0;
    h.ef_construction = params.ef_construction;
    h.ef_search = params.ef_search;
    h.entry_point = links.entry_point();
    h.max_level = links.max_level();
    out.write(reinterpret_cast<const char*>(&h), sizeof(h));
    auto flat = store.export_vectors();
    std::uint64_t n = flat.size();
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(flat.data()), static_cast<std::streamsize>(n * sizeof(float)));
    write_links(out, links, static_cast<std::size_t>(h.count));
    return out.good();
}

bool persist_load(
    const std::string& path,
    HnswParams& params,
    VectorStore& store,
    LinkStore& links) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    FileHeader h{};
    in.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (h.magic != kMagic) {
        return false;
    }
    params.dim = h.dim;
    params.M = h.M;
    params.M_max0 = h.M_max0;
    params.ef_construction = h.ef_construction;
    params.ef_search = h.ef_search;
    std::uint64_t n = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    std::vector<float> flat(static_cast<std::size_t>(n));
    in.read(reinterpret_cast<char*>(flat.data()), static_cast<std::streamsize>(n * sizeof(float)));
    store.import_vectors(flat, h.count);
    read_links(in, links, h.count);
    links.restore_state(h.entry_point, h.max_level);
    return in.good();
}

}