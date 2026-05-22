#pragma once

#include <string>
#include <vector>
#include <mutex>

#include "ragdb/types.hpp"
#include "ragdb/vector_store.hpp"
#include "ragdb/link_store.hpp"
#include "ragdb/hnsw_search.hpp"
#include "ragdb/hnsw_insert.hpp"

namespace ragdb {

class HnswIndex {
public:
    explicit HnswIndex(const HnswParams& params);

    Id add(const float* vec);

    void clear();

    std::vector<SearchHit> search(const float* query, std::size_t k) const;

    std::size_t size() const;

    bool save(const std::string& path) const;

    bool load(const std::string& path);

    const HnswParams& params() const;

private:
    HnswParams params_;
    VectorStore store_;
    LinkStore links_;
    HnswSearch search_;
    HnswInsert inserter_;
    mutable std::mutex mutex_;
};

}