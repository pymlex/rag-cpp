#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "rag_db/config.hpp"
#include "rag_db/types.hpp"


namespace rag_db {

struct NodeLinks {
    int level = 0;
    std::vector<std::vector<Label>> by_level;
};

class LinkStorage {
public:
    explicit LinkStorage(std::size_t m)
        : m_(m), m_max0_(m * 2) {}

    NodeLinks& node(Label id) {
        if (id >= nodes_.size()) {
            nodes_.resize(static_cast<std::size_t>(id) + 1);
        }
        return nodes_[static_cast<std::size_t>(id)];
    }

    const NodeLinks& node(Label id) const {
        return nodes_[static_cast<std::size_t>(id)];
    }

    void resize(std::size_t count) {
        nodes_.resize(count);
    }

    std::size_t max_neighbors(int level) const {
        return level == 0 ? m_max0_ : m_;
    }

    void connect(NodeLinks& node, Label other, int level) {
        auto& list = node.by_level[static_cast<std::size_t>(level)];
        if (std::find(list.begin(), list.end(), other) != list.end()) {
            return;
        }
        list.push_back(other);
        std::size_t cap = max_neighbors(level);
        if (list.size() > cap) {
            prune_list(list, level);
        }
    }

    void init_levels(NodeLinks& node, int level) {
        node.level = level;
        node.by_level.assign(static_cast<std::size_t>(level) + 1, {});
    }

    const std::vector<NodeLinks>& all() const {
        return nodes_;
    }

    std::vector<NodeLinks>& all_mut() {
        return nodes_;
    }

private:
    void prune_list(std::vector<Label>& list, int level) {
        std::size_t cap = max_neighbors(level);
        if (list.size() <= cap) {
            return;
        }
        list.resize(cap);
    }

    std::size_t m_;
    std::size_t m_max0_;
    std::vector<NodeLinks> nodes_;
};

}