#pragma once

#include <mutex>
#include <string>
#include <vector>

#include "rag_db/config.hpp"
#include "rag_db/hnsw_algorithm.hpp"
#include "rag_db/level_generator.hpp"
#include "rag_db/link_storage.hpp"
#include "rag_db/types.hpp"
#include "rag_db/vector_storage.hpp"
#include "rag_db/visited_pool.hpp"


namespace rag_db {

class HnswIndex {
public:
    HnswIndex(
        std::size_t dim,
        std::size_t max_elements,
        std::size_t m = kDefaultM,
        std::size_t ef_construction = kDefaultEfConstruction,
        bool cosine = true
    );

    Label add(const float* vector);
    void mark_deleted(Label label);
    std::vector<SearchHit> search(const float* query, std::size_t k, std::size_t ef) const;
    std::size_t size() const;
    std::size_t dim() const;
    Label entry_point() const;
    int max_level() const;
    bool save(const std::string& path) const;
    bool load(const std::string& path);

private:
    Label next_id();
    void connect_bidirectional(Label src, Label dst, int level);
    void shrink_neighbor_lists(Label id, int level);
    Label search_best_entry(const float* query) const;

    std::size_t dim_;
    std::size_t max_elements_;
    std::size_t m_;
    std::size_t ef_construction_;
    std::size_t ef_search_;
    bool cosine_;
    Label entry_point_;
    int max_level_;
    Label next_label_;
    LevelGenerator level_gen_;
    VectorStorage vectors_;
    LinkStorage links_;
    mutable VisitedPool visited_;
    mutable std::mutex mutex_;
};

}