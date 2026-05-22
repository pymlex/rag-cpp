#include "ragdb/hnsw_index.hpp"

#include "ragdb/persistence.hpp"

namespace ragdb {

HnswIndex::HnswIndex(const HnswParams& params)
    : params_(params),
      store_(params),
      links_(params),
      search_(params, store_, links_),
      inserter_(params, store_, links_, search_) {}

Id HnswIndex::add(const float* vec) {
    std::lock_guard<std::mutex> lock(mutex_);
    return inserter_.insert(vec);
}

void HnswIndex::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    store_.clear();
    links_.clear();
}

std::vector<SearchHit> HnswIndex::search(const float* query, std::size_t k) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return search_.knn(query, k, params_.ef_search);
}

std::size_t HnswIndex::size() const {
    return store_.count();
}

bool HnswIndex::save(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return persist_save(path, params_, store_, links_);
}

bool HnswIndex::load(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    return persist_load(path, params_, store_, links_);
}

const HnswParams& HnswIndex::params() const {
    return params_;
}

}