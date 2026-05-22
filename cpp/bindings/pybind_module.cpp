#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "rag_db/hnsw_index.hpp"

namespace py = pybind11;


PYBIND11_MODULE(rag_db_native, m) {
    m.doc() = "RAG HNSW vector database";

    py::class_<rag_db::HnswIndex>(m, "HnswIndex")
        .def(
            py::init<std::size_t, std::size_t, std::size_t, std::size_t, bool>(),
            py::arg("dim"),
            py::arg("max_elements"),
            py::arg("m") = rag_db::kDefaultM,
            py::arg("ef_construction") = rag_db::kDefaultEfConstruction,
            py::arg("cosine") = true
        )
        .def(
            "add",
            [](rag_db::HnswIndex& self, py::array_t<float> vec) {
                auto buf = vec.request();
                if (buf.ndim != 1) {
                    throw std::runtime_error("vector must be 1-D");
                }
                const float* data = static_cast<const float*>(buf.ptr);
                return static_cast<std::uint64_t>(self.add(data));
            },
            py::arg("vector")
        )
        .def("mark_deleted", &rag_db::HnswIndex::mark_deleted)
        .def(
            "search",
            [](rag_db::HnswIndex& self, py::array_t<float> query, std::size_t k, std::size_t ef) {
                auto buf = query.request();
                const float* data = static_cast<const float*>(buf.ptr);
                auto hits = self.search(data, k, ef);
                py::list labels;
                py::list distances;
                for (const auto& hit : hits) {
                    labels.append(static_cast<std::uint64_t>(hit.label));
                    distances.append(hit.distance);
                }
                return py::make_tuple(labels, distances);
            },
            py::arg("query"),
            py::arg("k"),
            py::arg("ef") = rag_db::kDefaultEfSearch
        )
        .def("save", &rag_db::HnswIndex::save)
        .def("load", &rag_db::HnswIndex::load)
        .def("size", &rag_db::HnswIndex::size)
        .def("dim", &rag_db::HnswIndex::dim);
}