#pragma once

#include <cstddef>
#include <vector>
#include <mutex>
#include <random>

#include "ragdb/types.hpp"

namespace ragdb {

class LinkStore {
public:
    explicit LinkStore(const HnswParams& params);

    Level level(Id id) const;

    void set_level(Id id, Level lvl);

    const std::vector<Id>& neighbors(Id id, Level layer) const;

    std::vector<Id>& neighbors_mut(Id id, Level layer);

    Id entry_point() const;

    Level max_level() const;

    void set_entry(Id id, Level lvl);

    void restore_state(Id entry, Level max_lvl);

    void clear();

    Level random_level(std::mt19937& rng);

    std::size_t max_links(Level layer) const;

private:
    HnswParams params_;
    std::vector<Level> levels_;
    std::vector<std::vector<std::vector<Id>>> links_;
    Id entry_;
    Level max_level_;
    mutable std::mutex mutex_;
};

}