#pragma once

#include <cmath>
#include <cstddef>


namespace rag_db {

constexpr std::size_t kDefaultM = 16;
constexpr std::size_t kDefaultEfConstruction = 200;
constexpr std::size_t kDefaultEfSearch = 64;
constexpr std::size_t kDefaultMaxElements = 200000;
constexpr float kDefaultMl = 1.0f / std::log(static_cast<float>(kDefaultM));
constexpr std::size_t kMaxLevelCap = 16;

}