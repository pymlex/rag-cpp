#pragma once

#include <vector>

#include "ragdb/types.hpp"
#include "ragdb/vector_store.hpp"
#include "ragdb/link_store.hpp"

namespace ragdb {

std::vector<Id> select_neighbors(
    const HnswParams& params,
    const VectorStore& store,
    Id query_id,
    const float* query,
    std::vector<SearchHit>& candidates,
    std::size_t max_neighbors);

void connect_bidirectional(
    LinkStore& links,
    VectorStore& store,
    const HnswParams& params,
    Id src,
    Id dst,
    Level layer);

void shrink_neighbors(
    LinkStore& links,
    VectorStore& store,
    const HnswParams& params,
    Id node,
    Level layer);

}