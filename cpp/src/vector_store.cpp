#include "ragdb/vector_store.hpp"

#include "ragdb/distance.hpp"

namespace ragdb {

VectorStore::VectorStore(const HnswParams& params)
    : params_(params),
      storage_(params.max_elements * params.dim, 0.0f),
      alive_(params.max_elements, false),
      next_id_(0),
      active_(0) {
    alive_.assign(params.max_elements, false);
}

Id VectorStore::add(const float* vec) {
    std::lock_guard<std::mutex> lock(mutex_);
    Id id = next_id_++;
    if (static_cast<std::size_t>(id) >= params_.max_elements) {
        return kInvalidId;
    }
    float* dst = storage_.data() + static_cast<std::size_t>(id) * params_.dim;
    for (std::size_t i = 0; i < params_.dim; ++i) {
        dst[i] = vec[i];
    }
    alive_[static_cast<std::size_t>(id)] = true;
    ++active_;
    return id;
}

void VectorStore::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    next_id_ = 0;
    active_ = 0;
    std::fill(alive_.begin(), alive_.end(), false);
    std::fill(storage_.begin(), storage_.end(), 0.0f);
}

bool VectorStore::is_active(Id id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= alive_.size()) {
        return false;
    }
    return alive_[static_cast<std::size_t>(id)];
}

const float* VectorStore::data(Id id) const {
    return storage_.data() + static_cast<std::size_t>(id) * params_.dim;
}

std::size_t VectorStore::dim() const {
    return params_.dim;
}

std::size_t VectorStore::count() const {
    return active_;
}

void VectorStore::reserve(std::size_t n) {
    if (n > params_.max_elements) {
        params_.max_elements = n;
        storage_.resize(n * params_.dim);
        alive_.resize(n, false);
    }
}

std::vector<float> VectorStore::export_vectors() const {
    return storage_;
}

void VectorStore::import_vectors(const std::vector<float>& flat, std::size_t count) {
    storage_ = flat;
    alive_.assign(count, true);
    next_id_ = static_cast<Id>(count);
    active_ = count;
}

Id VectorStore::max_id() const {
    return next_id_;
}

const std::vector<bool>& VectorStore::alive_flags() const {
    return alive_;
}

}