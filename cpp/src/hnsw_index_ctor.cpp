#include "rag_db/hnsw_index.hpp"


namespace rag_db {

HnswIndex::HnswIndex(
    std::size_t dim,
    std::size_t max_elements,
    std::size_t m,
    std::size_t ef_construction,
    bool cosine
)
    : dim_(dim),
      max_elements_(max_elements),
      m_(m),
      ef_construction_(ef_construction),
      ef_search_(kDefaultEfSearch),
      cosine_(cosine),
      entry_point_(0),
      max_level_(0),
      next_label_(0),
      level_gen_(kDefaultMl, 42),
      vectors_(dim, max_elements),
      links_(m),
      visited_(max_elements) {
    links_.resize(max_elements);
}

}