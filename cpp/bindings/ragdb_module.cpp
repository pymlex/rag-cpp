#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "ragdb/hnsw_index.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ragdb_native, m) {
    m.doc() = "RAG native HNSW vector index";

    py::class_<ragdb::SearchHit>(m, "SearchHit")
        .def_readonly("id", &ragdb::SearchHit::id)
        .def_readonly("distance", &ragdb::SearchHit::distance);

    py::class_<ragdb::HnswParams>(m, "HnswParams")
        .def(py::init<>())
        .def_readwrite("dim", &ragdb::HnswParams::dim)
        .def_readwrite("max_elements", &ragdb::HnswParams::max_elements)
        .def_readwrite("M", &ragdb::HnswParams::M)
        .def_readwrite("M_max0", &ragdb::HnswParams::M_max0)
        .def_readwrite("ef_construction", &ragdb::HnswParams::ef_construction)
        .def_readwrite("ef_search", &ragdb::HnswParams::ef_search);

    py::class_<ragdb::HnswIndex>(m, "HnswIndex")
        .def(py::init<ragdb::HnswParams>())
        .def("add", [](ragdb::HnswIndex& idx, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("expected 1-D float vector");
            }
            if (static_cast<std::size_t>(buf.size) != idx.params().dim) {
                throw std::runtime_error("vector dimension mismatch");
            }
            return idx.add(static_cast<float*>(buf.ptr));
        })
        .def("clear", &ragdb::HnswIndex::clear)
        .def("search", [](const ragdb::HnswIndex& idx, py::array_t<float> arr, std::size_t k) {
            auto buf = arr.request();
            auto hits = idx.search(static_cast<float*>(buf.ptr), k);
            py::list out;
            for (const auto& h : hits) {
                out.append(h);
            }
            return out;
        })
        .def("size", &ragdb::HnswIndex::size)
        .def("save", &ragdb::HnswIndex::save)
        .def("load", &ragdb::HnswIndex::load);
}