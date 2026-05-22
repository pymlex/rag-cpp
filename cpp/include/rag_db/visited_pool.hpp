#pragma once

#include <cstdint>
#include <vector>


namespace rag_db {

class VisitedPool {
public:
    explicit VisitedPool(std::size_t capacity)
        : tags_(capacity, 0), stamp_(1) {}

    void reset() {
        ++stamp_;
        if (stamp_ == 0) {
            std::fill(tags_.begin(), tags_.end(), 0);
            stamp_ = 1;
        }
    }

    bool visit(std::uint32_t id) {
        if (id >= tags_.size()) {
            return false;
        }
        if (tags_[id] == stamp_) {
            return false;
        }
        tags_[id] = stamp_;
        return true;
    }

    void resize(std::size_t capacity) {
        tags_.assign(capacity, 0);
        stamp_ = 1;
    }

private:
    std::vector<std::uint32_t> tags_;
    std::uint32_t stamp_;
};

}