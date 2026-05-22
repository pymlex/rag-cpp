#pragma once

#include <vector>
#include <random>

#include "ragdb/types.hpp"
#include "ragdb/vector_store.hpp"
#include "ragdb/link_store.hpp"
#include "ragdb/visited_set.hpp"
#include "ragdb/candidate_queue.hpp"

namespace ragdb {

class HnswSearch {
public:
    HnswSearch(const HnswParams& params, VectorStore& store, LinkStore& links);

    std::vector<SearchHit> knn(const float* query, std::size_t k, std::int32_t ef) const;

    Id greedy_closest(const float* query, Id start, Level layer) const;

    void search_layer(
        const float* query,
        Id entry,
        std::int32_t ef,
        Level layer,
        CandidateMaxHeap& result) const;

    HnswParams params_;
    VectorStore& store_;
    LinkStore& links_;
};

}