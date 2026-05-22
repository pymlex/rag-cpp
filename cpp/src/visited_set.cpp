#include "ragdb/visited_set.hpp"

namespace ragdb {

VisitedSet::VisitedSet(std::size_t capacity)
    : stamps_(capacity, 0),
      tags_(capacity, 0),
      generation_(1) {}

bool VisitedSet::mark(Id id) {
    if (id < 0 || static_cast<std::size_t>(id) >= stamps_.size()) {
        return false;
    }
    const std::size_t idx = static_cast<std::size_t>(id);
    if (tags_[idx] == generation_) {
        return false;
    }
    tags_[idx] = generation_;
    return true;
}

void VisitedSet::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++generation_;
    if (generation_ == 0) {
        std::fill(tags_.begin(), tags_.end(), 0);
        generation_ = 1;
    }
}

}