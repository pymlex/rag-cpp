#include "ragdb/hnsw_insert.hpp"

#include "ragdb/neighbor_select.hpp"
#include "ragdb/candidate_queue.hpp"

namespace ragdb {

HnswInsert::HnswInsert(
    const HnswParams& params,
    VectorStore& store,
    LinkStore& links,
    HnswSearch& search)
    : params_(params),
      store_(store),
      links_(links),
      search_(search),
      rng_(std::random_device{}()) {}

Id HnswInsert::insert(const float* vec) {
    std::lock_guard<std::mutex> lock(insert_mutex_);
    Id id = store_.add(vec);
    if (id == kInvalidId) {
        return kInvalidId;
    }
    Level lvl = links_.random_level(rng_);
    links_.set_level(id, lvl);
    if (links_.entry_point() == kInvalidId) {
        links_.set_entry(id, lvl);
        return id;
    }
    Id ep = links_.entry_point();
    const float* q = store_.data(id);
    if (lvl > links_.max_level()) {
        for (Level layer = links_.max_level() + 1; layer <= lvl; ++layer) {
            ep = search_.greedy_closest(q, ep, layer);
        }
        links_.set_entry(id, lvl);
    }
    for (Level layer = std::min(lvl, links_.max_level()); layer >= 0; --layer) {
        ep = search_.greedy_closest(q, ep, layer);
        CandidateMaxHeap heap;
        search_.search_layer(q, ep, params_.ef_construction, layer, heap);
        std::vector<SearchHit> cands;
        while (!heap.empty()) {
            cands.push_back(heap.pop());
        }
        std::sort(cands.begin(), cands.end(),
            [](const SearchHit& a, const SearchHit& b) { return a.distance < b.distance; });
        std::size_t max_n = links_.max_links(layer);
        std::vector<Id> chosen = select_neighbors(params_, store_, id, q, cands, max_n);
        for (Id nb : chosen) {
            connect_bidirectional(links_, store_, params_, id, nb, layer);
        }
        ep = chosen.empty() ? ep : chosen.front();
    }
    return id;
}

}