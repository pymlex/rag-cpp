#include "rag_db/hnsw_index.hpp"


namespace rag_db {

Label HnswIndex::search_best_entry(const float* query) const {
    Label curr = entry_point_;
    if (!vectors_.is_alive(curr)) {
        return curr;
    }
    for (int lc = max_level_; lc > 0; --lc) {
        bool changed = true;
        while (changed) {
            changed = false;
            const auto& node = links_.node(curr);
            if (static_cast<int>(node.by_level.size()) <= lc) {
                continue;
            }
            float curr_dist = cosine_
                ? cosine_distance(query, vectors_.vector(curr), dim_)
                : l2_squared(query, vectors_.vector(curr), dim_);
            for (Label nb : node.by_level[static_cast<std::size_t>(lc)]) {
                if (!vectors_.is_alive(nb)) {
                    continue;
                }
                float d = cosine_
                    ? cosine_distance(query, vectors_.vector(nb), dim_)
                    : l2_squared(query, vectors_.vector(nb), dim_);
                if (d < curr_dist) {
                    curr_dist = d;
                    curr = nb;
                    changed = true;
                }
            }
        }
    }
    return curr;
}


std::vector<SearchHit> HnswIndex::search(const float* query, std::size_t k, std::size_t ef) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (next_label_ == 0) {
        return {};
    }
    std::size_t use_ef = std::max(ef, k);
    Label curr = search_best_entry(query);
    auto found = search_layer(vectors_, links_, visited_, query, curr, 0, use_ef, cosine_);
    std::vector<SearchHit> hits;
    hits.reserve(k);
    for (std::size_t i = 0; i < found.size() && hits.size() < k; ++i) {
        if (!vectors_.is_alive(found[i].label)) {
            continue;
        }
        hits.push_back({found[i].label, found[i].distance});
    }
    return hits;
}


void HnswIndex::mark_deleted(Label label) {
    std::lock_guard<std::mutex> lock(mutex_);
    vectors_.mark_dead(label);
}


std::size_t HnswIndex::size() const {
    return next_label_;
}


std::size_t HnswIndex::dim() const {
    return dim_;
}


Label HnswIndex::entry_point() const {
    return entry_point_;
}


int HnswIndex::max_level() const {
    return max_level_;
}

}