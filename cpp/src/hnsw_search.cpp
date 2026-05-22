#include "ragdb/hnsw_search.hpp"

#include "ragdb/distance.hpp"

namespace ragdb {

HnswSearch::HnswSearch(const HnswParams& params, VectorStore& store, LinkStore& links)
    : params_(params), store_(store), links_(links) {}

Id HnswSearch::greedy_closest(const float* query, Id start, Level layer) const {
    Id curr = start;
    float curr_dist = cosine_distance(query, store_.data(curr), params_.dim);
    bool changed = true;
    while (changed) {
        changed = false;
        const auto& nbrs = links_.neighbors(curr, layer);
        for (Id cand : nbrs) {
            if (!store_.is_active(cand)) {
                continue;
            }
            float d = cosine_distance(query, store_.data(cand), params_.dim);
            if (d < curr_dist) {
                curr_dist = d;
                curr = cand;
                changed = true;
            }
        }
    }
    return curr;
}

void HnswSearch::search_layer(
    const float* query,
    Id entry,
    std::int32_t ef,
    Level layer,
    CandidateMaxHeap& result) const {
    VisitedSet visited(params_.max_elements);
    CandidateMinHeap candidates;
    float entry_dist = cosine_distance(query, store_.data(entry), params_.dim);
    candidates.push(entry, entry_dist);
    result.push(entry, entry_dist);
    visited.mark(entry);
    while (!candidates.empty()) {
        SearchHit c = candidates.pop();
        if (c.distance > result.top().distance && result.size() >= static_cast<std::size_t>(ef)) {
            break;
        }
        const auto& nbrs = links_.neighbors(c.id, layer);
        for (Id nid : nbrs) {
            if (!store_.is_active(nid) || !visited.mark(nid)) {
                continue;
            }
            float d = cosine_distance(query, store_.data(nid), params_.dim);
            if (result.size() < static_cast<std::size_t>(ef) || d < result.top().distance) {
                result.push(nid, d);
                candidates.push(nid, d);
                if (result.size() > static_cast<std::size_t>(ef)) {
                    result.pop();
                }
            }
        }
    }
}

std::vector<SearchHit> HnswSearch::knn(const float* query, std::size_t k, std::int32_t ef) const {
    if (links_.entry_point() == kInvalidId) {
        return {};
    }
    Id curr = links_.entry_point();
    for (Level layer = links_.max_level(); layer > 0; --layer) {
        curr = greedy_closest(query, curr, layer);
    }
    CandidateMaxHeap result;
    search_layer(query, curr, ef, 0, result);
    std::vector<SearchHit> hits;
    while (!result.empty() && hits.size() < k) {
        hits.push_back(result.pop());
    }
    std::sort(hits.begin(), hits.end(),
        [](const SearchHit& a, const SearchHit& b) { return a.distance < b.distance; });
    return hits;
}

}