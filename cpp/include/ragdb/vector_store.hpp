#pragma once

#include <cstddef>
#include <vector>
#include <mutex>

#include "ragdb/types.hpp"

namespace ragdb {

class VectorStore {
public:
    explicit VectorStore(const HnswParams& params);

    Id add(const float* vec);

    void clear();

    bool is_active(Id id) const;

    const float* data(Id id) const;

    std::size_t dim() const;

    std::size_t count() const;

    void reserve(std::size_t n);

    std::vector<float> export_vectors() const;

    void import_vectors(const std::vector<float>& flat, std::size_t count);

    Id max_id() const;

    const std::vector<bool>& alive_flags() const;

private:
    HnswParams params_;
    std::vector<float> storage_;
    std::vector<bool> alive_;
    Id next_id_;
    std::size_t active_;
    mutable std::mutex mutex_;
};

}