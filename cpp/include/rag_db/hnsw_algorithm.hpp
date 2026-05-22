#pragma once

#include <algorithm>
#include <limits>
#include <random>

#include "rag_db/distance.hpp"
#include "rag_db/link_storage.hpp"
#include "rag_db/priority_queue.hpp"
#include "rag_db/types.hpp"
#include "rag_db/vector_storage.hpp"
#include "rag_db/visited_pool.hpp"


namespace rag_db {

inline float node_distance(
    const VectorStorage& storage,
    Label a,
    Label b,
    bool cosine
) {
    const float* va = storage.vector(a);
    const float* vb = storage.vector(b);
    std::size_t dim = storage.dim();
    if (cosine) {
        return cosine_distance(va, vb, dim);
    }
    return l2_squared(va, vb, dim);
}


inline std::vector<Neighbor> search_layer(
    const VectorStorage& storage,
    const LinkStorage& links,
    VisitedPool& visited,
    const float* query,
    Label entry,
    int level,
    std::size_t ef,
    bool cosine
) {
    CandidateMaxHeap candidates;
    candidates.set_limit(ef);
    CandidateMaxHeap results;
    results.set_limit(ef);

    visited.reset();
    float entry_dist = cosine
        ? cosine_distance(query, storage.vector(entry), storage.dim())
        : l2_squared(query, storage.vector(entry), storage.dim());

    candidates.push({entry, entry_dist});
    results.push({entry, entry_dist});
    visited.visit(static_cast<std::uint32_t>(entry));

    std::vector<Neighbor> frontier = candidates.sorted();
    std::size_t idx = 0;
    while (idx < frontier.size()) {
        Neighbor current = frontier[idx++];
        const auto& node = links.node(current.label);
        if (static_cast<int>(node.by_level.size()) <= level) {
            continue;
        }
        const auto& neighbors = node.by_level[static_cast<std::size_t>(level)];
        for (Label nb : neighbors) {
            if (!storage.is_alive(nb)) {
                continue;
            }
            if (!visited.visit(static_cast<std::uint32_t>(nb))) {
                continue;
            }
            float dist = cosine
                ? cosine_distance(query, storage.vector(nb), storage.dim())
                : l2_squared(query, storage.vector(nb), storage.dim());
            candidates.push({nb, dist});
            results.push({nb, dist});
            if (dist < current.distance) {
                frontier.push_back({nb, dist});
            }
        }
    }
    return results.sorted();
}


inline std::vector<Neighbor> select_neighbors(
    const VectorStorage& storage,
    const std::vector<Neighbor>& candidates,
    std::size_t m,
    bool cosine
) {
    std::vector<Neighbor> sorted = candidates;
    std::sort(sorted.begin(), sorted.end(), [](const Neighbor& a, const Neighbor& b) {
        return a.distance < b.distance;
    });
    if (sorted.size() <= m) {
        return sorted;
    }
    std::vector<Neighbor> selected;
    selected.reserve(m);
    for (const auto& cand : sorted) {
        bool bad = false;
        for (const auto& pick : selected) {
            float d = node_distance(storage, cand.label, pick.label, cosine);
            if (d < cand.distance) {
                bad = true;
                break;
            }
        }
        if (!bad) {
            selected.push_back(cand);
        }
        if (selected.size() >= m) {
            break;
        }
    }
    if (selected.size() < m) {
        for (const auto& cand : sorted) {
            if (selected.size() >= m) {
                break;
            }
            if (std::find_if(selected.begin(), selected.end(), [&](const Neighbor& n) {
                    return n.label == cand.label;
                }) == selected.end()) {
                selected.push_back(cand);
            }
        }
    }
    return selected;
}

}