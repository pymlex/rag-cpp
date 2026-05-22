#pragma once

#include <cstddef>
#include <vector>

namespace ragdb {

float cosine_distance(const float* a, const float* b, std::size_t dim);

float inner_product(const float* a, const float* b, std::size_t dim);

void normalize_inplace(std::vector<float>& v, std::size_t dim);

}