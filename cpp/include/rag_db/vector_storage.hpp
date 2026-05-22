#pragma once

#include <cstddef>
#include <vector>

#include "rag_db/types.hpp"


namespace rag_db {

class VectorStorage {
public:
    VectorStorage(std::size_t dim, std::size_t capacity)
        : dim_(dim), data_(capacity * dim, 0.0f), alive_(capacity, 0) {}

    void resize_capacity(std::size_t capacity) {
        data_.resize(capacity * dim_, 0.0f);
        alive_.resize(capacity, 0);
    }

    std::size_t dim() const {
        return dim_;
    }

    const float* vector(Label id) const {
        return data_.data() + static_cast<std::size_t>(id) * dim_;
    }

    float* vector_mut(Label id) {
        return data_.data() + static_cast<std::size_t>(id) * dim_;
    }

    void set(Label id, const float* src) {
        float* dst = vector_mut(id);
        for (std::size_t i = 0; i < dim_; ++i) {
            dst[i] = src[i];
        }
        alive_[static_cast<std::size_t>(id)] = 1;
    }

    bool is_alive(Label id) const {
        return alive_[static_cast<std::size_t>(id)] != 0;
    }

    void mark_dead(Label id) {
        alive_[static_cast<std::size_t>(id)] = 0;
    }

    const std::vector<float>& raw() const {
        return data_;
    }

    const std::vector<std::uint8_t>& alive_flags() const {
        return alive_;
    }

    void load_raw(const std::vector<float>& data, const std::vector<std::uint8_t>& alive) {
        data_ = data;
        alive_ = alive;
    }

private:
    std::size_t dim_;
    std::vector<float> data_;
    std::vector<std::uint8_t> alive_;
};

}