#include "ragdb/link_store.hpp"

#include <cmath>

namespace ragdb {

LinkStore::LinkStore(const HnswParams& params)
    : params_(params),
      levels_(params.max_elements, 0),
      links_(params.max_elements),
      entry_(kInvalidId),
      max_level_(0) {}

Level LinkStore::level(Id id) const {
    return levels_[static_cast<std::size_t>(id)];
}

void LinkStore::set_level(Id id, Level lvl) {
    levels_[static_cast<std::size_t>(id)] = lvl;
    auto& node_links = links_[static_cast<std::size_t>(id)];
    node_links.resize(static_cast<std::size_t>(lvl) + 1);
}

const std::vector<Id>& LinkStore::neighbors(Id id, Level layer) const {
    return links_[static_cast<std::size_t>(id)][static_cast<std::size_t>(layer)];
}

std::vector<Id>& LinkStore::neighbors_mut(Id id, Level layer) {
    return links_[static_cast<std::size_t>(id)][static_cast<std::size_t>(layer)];
}

Id LinkStore::entry_point() const {
    return entry_;
}

Level LinkStore::max_level() const {
    return max_level_;
}

void LinkStore::set_entry(Id id, Level lvl) {
    entry_ = id;
    if (lvl > max_level_) {
        max_level_ = lvl;
    }
}

void LinkStore::restore_state(Id entry, Level max_lvl) {
    entry_ = entry;
    max_level_ = max_lvl;
}

void LinkStore::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    levels_.assign(params_.max_elements, 0);
    links_.assign(params_.max_elements, {});
    entry_ = kInvalidId;
    max_level_ = 0;
}

Level LinkStore::random_level(std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    Level lvl = 0;
    while (dist(rng) < params_.level_mult && lvl < 16) {
        ++lvl;
    }
    return lvl;
}

std::size_t LinkStore::max_links(Level layer) const {
    if (layer == 0) {
        return static_cast<std::size_t>(params_.M_max0);
    }
    return static_cast<std::size_t>(params_.M);
}

}