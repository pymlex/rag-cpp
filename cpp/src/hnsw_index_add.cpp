#include "rag_db/hnsw_index.hpp"


namespace rag_db {

Label HnswIndex::next_id() {
    return next_label_++;
}


void HnswIndex::connect_bidirectional(Label src, Label dst, int level) {
    links_.connect(links_.node(src), dst, level);
    links_.connect(links_.node(dst), src, level);
}


void HnswIndex::shrink_neighbor_lists(Label id, int level) {
    auto& node = links_.node(id);
    auto& list = node.by_level[static_cast<std::size_t>(level)];
    if (list.empty()) {
        return;
    }
    std::vector<Neighbor> scored;
    scored.reserve(list.size());
    for (Label nb : list) {
        if (!vectors_.is_alive(nb)) {
            continue;
        }
        scored.push_back({nb, node_distance(vectors_, id, nb, cosine_)});
    }
    auto picked = select_neighbors(vectors_, scored, links_.max_neighbors(level), cosine_);
    list.clear();
    for (const auto& n : picked) {
        list.push_back(n.label);
    }
}


Label HnswIndex::add(const float* vector) {
    std::lock_guard<std::mutex> lock(mutex_);
    Label id = next_id();
    if (static_cast<std::size_t>(id) >= max_elements_) {
        return id;
    }
    vectors_.set(id, vector);
    int level = level_gen_.sample_level();
    links_.init_levels(links_.node(id), level);
    if (next_label_ == 1) {
        entry_point_ = id;
        max_level_ = level;
        return id;
    }
    Label curr_entry = search_best_entry(vector);
    for (int lc = max_level_; lc > level; --lc) {
        auto layer = search_layer(vectors_, links_, visited_, vector, curr_entry, lc, 1, cosine_);
        if (!layer.empty()) {
            curr_entry = layer.front().label;
        }
    }
    for (int lc = std::min(level, max_level_); lc >= 0; --lc) {
        auto found = search_layer(
            vectors_, links_, visited_, vector, curr_entry, lc, ef_construction_, cosine_
        );
        auto picked = select_neighbors(vectors_, found, m_, cosine_);
        for (const auto& n : picked) {
            connect_bidirectional(id, n.label, lc);
        }
        for (const auto& n : picked) {
            shrink_neighbor_lists(n.label, lc);
        }
    }
    if (level > max_level_) {
        max_level_ = level;
        entry_point_ = id;
    }
    return id;
}

}