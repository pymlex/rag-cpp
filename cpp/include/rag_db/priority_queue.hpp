#pragma once

#include <algorithm>
#include <queue>
#include <vector>

#include "rag_db/types.hpp"


namespace rag_db {

class CandidateMaxHeap {
public:
    void push(const Neighbor& n) {
        if (items_.size() < limit_) {
            items_.push_back(n);
            std::push_heap(items_.begin(), items_.end(), cmp_);
            if (items_.size() == limit_) {
                worst_ = items_.front().distance;
            }
            return;
        }
        if (n.distance >= worst_) {
            return;
        }
        std::pop_heap(items_.begin(), items_.end(), cmp_);
        items_.back() = n;
        std::push_heap(items_.begin(), items_.end(), cmp_);
        worst_ = items_.front().distance;
    }

    void set_limit(std::size_t limit) {
        limit_ = limit;
        items_.clear();
        worst_ = 1e30f;
    }

    const std::vector<Neighbor>& data() const {
        return items_;
    }

    std::vector<Neighbor> sorted() const {
        std::vector<Neighbor> out = items_;
        std::sort(out.begin(), out.end(), [](const Neighbor& a, const Neighbor& b) {
            return a.distance < b.distance;
        });
        return out;
    }

private:
    static bool cmp_(const Neighbor& a, const Neighbor& b) {
        return a.distance < b.distance;
    }

    std::size_t limit_ = 0;
    float worst_ = 1e30f;
    std::vector<Neighbor> items_;
};

}