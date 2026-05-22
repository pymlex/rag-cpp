#pragma once

#include <mutex>
#include <random>

#include "ragdb/types.hpp"
#include "ragdb/vector_store.hpp"
#include "ragdb/link_store.hpp"
#include "ragdb/hnsw_search.hpp"

namespace ragdb {

class HnswInsert {
public:
    HnswInsert(
        const HnswParams& params,
        VectorStore& store,
        LinkStore& links,
        HnswSearch& search);

    Id insert(const float* vec);

private:
    HnswParams params_;
    VectorStore& store_;
    LinkStore& links_;
    HnswSearch& search_;
    std::mutex insert_mutex_;
    std::mt19937 rng_;
};

}