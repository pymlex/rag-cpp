#include "rag_db/c_api.h"

#include <cstring>
#include <memory>
#include <vector>

#include "rag_db/hnsw_index.hpp"


struct RagDbHandle {
    rag_db::HnswIndex index;
};

extern "C" {

RagDbHandle* rag_db_create(
    size_t dim,
    size_t max_elements,
    size_t m,
    size_t ef_construction,
    int cosine
) {
    return new RagDbHandle{rag_db::HnswIndex(
        dim,
        max_elements,
        m,
        ef_construction,
        cosine != 0
    )};
}

void rag_db_destroy(RagDbHandle* handle) {
    delete handle;
}

uint64_t rag_db_add(RagDbHandle* handle, const float* vector) {
    return static_cast<uint64_t>(handle->index.add(vector));
}

void rag_db_mark_deleted(RagDbHandle* handle, uint64_t label) {
    handle->index.mark_deleted(static_cast<rag_db::Label>(label));
}

size_t rag_db_search(
    RagDbHandle* handle,
    const float* query,
    size_t k,
    size_t ef,
    uint64_t* out_labels,
    float* out_distances
) {
    auto hits = handle->index.search(query, k, ef);
    for (size_t i = 0; i < hits.size(); ++i) {
        out_labels[i] = static_cast<uint64_t>(hits[i].label);
        out_distances[i] = hits[i].distance;
    }
    return hits.size();
}

int rag_db_save(RagDbHandle* handle, const char* path) {
    return handle->index.save(path) ? 1 : 0;
}

int rag_db_load(RagDbHandle* handle, const char* path) {
    return handle->index.load(path) ? 1 : 0;
}

size_t rag_db_dim(RagDbHandle* handle) {
    return handle->index.dim();
}

uint64_t rag_db_size(RagDbHandle* handle) {
    return static_cast<uint64_t>(handle->index.size());
}

}