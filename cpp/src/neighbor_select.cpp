#include "ragdb/neighbor_select.hpp"

#include "ragdb/link_store.hpp"
#include "ragdb/distance.hpp"

#include <algorithm>

namespace ragdb {

std::vector<Id> select_neighbors(
    const HnswParams& params,
    const VectorStore& store,
    Id query_id,
    const float* query,
    std::vector<SearchHit>& candidates,
    std::size_t max_neighbors) {
    std::sort(candidates.begin(), candidates.end(),
        [](const SearchHit& a, const SearchHit& b) { return a.distance < b.distance; });
    std::vector<Id> selected;
    selected.reserve(max_neighbors);
    for (const auto& c : candidates) {
        if (c.id == query_id) {
            continue;
        }
        bool bad = false;
        for (Id s : selected) {
            float d = cosine_distance(store.data(c.id), store.data(s), params.dim);
            if (d < c.distance) {
                bad = true;
                break;
            }
        }
        if (!bad) {
            selected.push_back(c.id);
        }
        if (selected.size() >= max_neighbors) {
            break;
        }
    }
    return selected;
}

void connect_bidirectional(
    LinkStore& links,
    VectorStore& store,
    const HnswParams& params,
    Id src,
    Id dst,
    Level layer) {
    if (!links.has_layer(src, layer) || !links.has_layer(dst, layer)) {
        return;
    }
    auto& a = links.neighbors_mut(src, layer);
    if (std::find(a.begin(), a.end(), dst) == a.end()) {
        a.push_back(dst);
    }
    auto& b = links.neighbors_mut(dst, layer);
    if (std::find(b.begin(), b.end(), src) == b.end()) {
        b.push_back(src);
    }
    shrink_neighbors(links, store, params, src, layer);
    shrink_neighbors(links, store, params, dst, layer);
}

void shrink_neighbors(
    LinkStore& links,
    VectorStore& store,
    const HnswParams& params,
    Id node,
    Level layer) {
    const std::size_t max_l = links.max_links(layer);
    auto& nbrs = links.neighbors_mut(node, layer);
    if (nbrs.size() <= max_l) {
        return;
    }
    const float* q = store.data(node);
    std::vector<SearchHit> cand;
    cand.reserve(nbrs.size());
    for (Id nid : nbrs) {
        if (!store.is_active(nid)) {
            continue;
        }
        cand.push_back({nid, cosine_distance(q, store.data(nid), params.dim)});
    }
    std::vector<Id> kept = select_neighbors(params, store, node, q, cand, max_l);
    nbrs = kept;
}

}