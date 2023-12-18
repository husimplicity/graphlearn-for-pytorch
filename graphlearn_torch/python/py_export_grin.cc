/* Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <torch/extension.h>

#include "graphlearn_torch/grin/grin_graph.h"
#include "graphlearn_torch/grin/grin_feature.h"
#include "graphlearn_torch/grin/grin_random_sampler.h"
#include "graphlearn_torch/grin/grin_sampler.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Python bindings for grin utils C++ frontend";

  py::class_<GrinGraph>(m, "CppGrinGraph")
    .def(py::init<const char*, const std::string&>())
    .def("get_row_count", &GrinGraph::GetRowCount)
    .def("get_edge_count", &GrinGraph::GetEdgeCount)
    .def("get_col_count", &GrinGraph::GetColCount)
    .def("get_src_type_name", &GrinGraph::GetSrcTypeName)
    .def("get_dst_type_name", &GrinGraph::GetDstTypeName)
    .def("get_edge_type_name", &GrinGraph::GetEdgeTypeName);
    
  py::class_<GrinRandomSampler>(m, "GrinRandomSampler")
    .def(py::init<GrinGraph*>())
    .def("sample", &GrinRandomSampler::Sample,
         py::arg("nodes"), py::arg("req_num"));

  py::class_<GrinVertexFeature>(m, "GrinVertexFeature")
    .def(py::init<const char*, const std::string&>())
    .def("cpu_get", &GrinVertexFeature::cpu_get,
         py::arg("ex_ids"))
    .def("get_labels", &GrinVertexFeature::get_labels,
         py::arg("ex_ids"));
}