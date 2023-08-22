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

#include "graphlearn_torch/csrc/cpu/inducer.h"
#include "graphlearn_torch/csrc/cpu/random_negative_sampler.h"
#include "graphlearn_torch/csrc/cpu/random_sampler.h"
#include "graphlearn_torch/csrc/cpu/subgraph_op.h"
#include "graphlearn_torch/include/graph.h"
#include "graphlearn_torch/include/negative_sampler.h"
#include "graphlearn_torch/include/sample_queue.h"
#include "graphlearn_torch/include/sampler.h"
#include "graphlearn_torch/include/stitch_sample_results.h"
#include "graphlearn_torch/include/types.h"
#include "graphlearn_torch/include/vineyard_utils.h"

#ifdef WITH_VINEYARD
#include "graphlearn_torch/include/grin/grin_graph.h"
#include "graphlearn_torch/include/grin/grin_feature.h"
#endif

#ifdef WITH_GART
#include "graphlearn_torch/include/grin/grin_graph.h"
#include "graphlearn_torch/include/grin/grin_feature.h"
#include "graphlearn_torch/csrc/cpu/grin/grin_random_sampler.h"
#include "graphlearn_torch/include/grin/grin_sampler.h"
#endif

#ifdef WITH_CUDA
#include "graphlearn_torch/csrc/cuda/inducer.cuh"
#include "graphlearn_torch/csrc/cuda/random_negative_sampler.cuh"
#include "graphlearn_torch/csrc/cuda/random_sampler.cuh"
#include "graphlearn_torch/csrc/cuda/subgraph_op.cuh"
#include "graphlearn_torch/include/unified_tensor.cuh"
#endif

namespace py = pybind11;
using namespace graphlearn_torch;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Python bindings for graphlearn_torch C++ frontend";
  m.def("cpu_stitch_sample_results", &CPUStitchSampleResults);
#ifdef WITH_CUDA
  m.def("cuda_stitch_sample_results", &CUDAStitchSampleResults);
#endif
#ifdef WITH_VINEYARD
  m.def("vineyard_to_csr", &ToCSR);
  m.def("load_vertex_feature_from_vineyard", &LoadVertexFeatures);
  m.def("load_edge_feature_from_vineyard", &LoadEdgeFeatures);
#endif

  py::enum_<DataType>(m, "DataType")
    .value("Int32", DataType::Int32)
    .value("Int64", DataType::Int64)
    .value("Float32", DataType::Float32)
    .value("Float64", DataType::Float64);

  py::enum_<GraphMode>(m, "GraphMode")
    .value("DMA", GraphMode::DMA)
    .value("ZERO_COPY", GraphMode::ZERO_COPY);

  py::class_<Graph>(m, "Graph")
    .def(py::init<>())
    .def("init_cpu_from_csr", &Graph::InitCPUGraphFromCSR,
         py::arg("indptr"), py::arg("indices"), py::arg("edge_ids"))
#ifdef WITH_CUDA
    .def("init_cuda_from_csr",
         py::overload_cast<const torch::Tensor&,
            const torch::Tensor&,
            int,
            GraphMode,
            const torch::Tensor&>(&Graph::InitCUDAGraphFromCSR),
         py::arg("indptr"), py::arg("indices"),
         py::arg("device"), py::arg("mode"), py::arg("edge_ids"))
#endif
    .def("get_row_count", &Graph::GetRowCount)
    .def("get_edge_count", &Graph::GetEdgeCount)
    .def("get_col_count", &Graph::GetColCount)
    .def("get_mode", &Graph::GetGraphMode);
  
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
         py::arg("ex_ids"), py::arg("num_props"))
    .def("get_labels", &GrinVertexFeature::get_labels,
         py::arg("ex_ids"));
  
  // py::class_<GrinEdgeFeature>(m, "GrinEdgeFeature")
  //   .def(py::init<const char*, const std::string&>())
  //   .def("cpu_get", &GrinEdgeFeature::cpu_get,
  //        py::arg("ex_ids"));

  py::class_<SubGraph>(m, "SubGraph")
    .def(py::init<>())
    .def_readwrite("nodes", &SubGraph::nodes)
    .def_readwrite("rows", &SubGraph::rows)
    .def_readwrite("cols", &SubGraph::cols)
    .def_readwrite("eids", &SubGraph::eids);

  py::class_<CPURandomSampler>(m, "CPURandomSampler")
    .def(py::init<const Graph*>())
    .def("sample", &CPURandomSampler::Sample,
         py::arg("ids"), py::arg("req_num"))
    .def("sample_with_edge", &CPURandomSampler::SampleWithEdge,
         py::arg("ids"), py::arg("req_num"));

  py::class_<CPURandomNegativeSampler>(m, "CPURandomNegativeSampler")
    .def(py::init<const Graph*>())
    .def("sample", &CPURandomNegativeSampler::Sample,
         py::arg("req_num"), py::arg("trials_num"), py::arg("padding"));

  py::class_<CPUInducer>(m, "CPUInducer")
    .def(py::init<int32_t>())
    .def("init_node", &CPUInducer::InitNode,
         py::arg("seed"))
    .def("induce_next", &CPUInducer::InduceNext,
         py::arg("srcs"), py::arg("nbrs"), py::arg("nbrs_num"));

  py::class_<CPUHeteroInducer>(m, "CPUHeteroInducer")
    .def(py::init<std::unordered_map<std::string, int32_t>>())
    .def("init_node", &CPUHeteroInducer::InitNode,
         py::arg("seed"))
    .def("induce_next", &CPUHeteroInducer::InduceNext,
         py::arg("hetero_nbrs"));

  py::class_<CPUSubGraphOp>(m, "CPUSubGraphOp")
    .def(py::init<const Graph*>())
    .def("node_subgraph", &CPUSubGraphOp::NodeSubGraph,
         py::arg("srcs"), py::arg("with_edge"));

  py::class_<SampleQueue>(m, "SampleQueue")
    .def(py::init<size_t, size_t>(), py::arg("capacity"), py::arg("buf_size"))
    .def("pin_memory", &SampleQueue::PinMemory)
    .def("send", &SampleQueue::Enqueue, py::arg("msg"),
         py::call_guard<py::gil_scoped_release>())
    .def("receive", &SampleQueue::Dequeue,
         py::call_guard<py::gil_scoped_release>())
    .def(py::pickle(
        [](const SampleQueue& q) { // __getstate__
            return q.ShmId();
        },
        [](int shmid) { // __setstate__
            /* Create a new C++ instance */
            return new SampleQueue{shmid};
        }
    ));

#ifdef WITH_CUDA
  py::class_<SharedTensor>(m, "SharedTensor")
    .def(py::init<>())
    .def("share_cuda_ipc", [](SharedTensor& self) {
      const auto& res = self.ShareCUDAIpc();
      auto handle = PyBytes_FromStringAndSize((char *)&(std::get<1>(res)),
                                              CUDA_IPC_HANDLE_SIZE);
      auto bytes_obj = py::reinterpret_steal<py::object>((PyObject *)handle);
      return std::make_tuple(std::get<0>(res), bytes_obj, std::get<2>(res));
    })
    .def("from_cuda_ipc",
      [](SharedTensor& self,
         const std::tuple<int32_t, std::string, std::vector<int64_t>>& ipc_data) {
        auto device = std::get<0>(ipc_data);
        auto shape = std::get<2>(ipc_data);
        auto handle = std::get<1>(ipc_data);
        auto ipc_handle =
          reinterpret_cast<const cudaIpcMemHandle_t *>(handle.c_str());
        self.FromCUDAIpc(std::make_tuple(device, *ipc_handle, shape));
      }, py::arg("cuda_ipc"));

  py::class_<UnifiedTensor>(m, "UnifiedTensor")
    .def(py::init<>([] (int32_t device, py::object py_dtype) {
      auto dtype = torch::python::detail::py_object_to_dtype(py_dtype);
      return new UnifiedTensor{device, dtype};
    }))
    .def("__getitem__", &UnifiedTensor::operator[])
    .def("append_shared_tensor", &UnifiedTensor::AppendSharedTensor,
      py::arg("shared_tensor"))
    .def("append_cpu_tensor", &UnifiedTensor::AppendCPUTensor,
      py::arg("cpu_tensor"))
    .def("init_from", &UnifiedTensor::InitFrom,
      py::arg("tensors"), py::arg("tensor_devices"))
    .def("share_cuda_ipc", &UnifiedTensor::ShareCUDAIpc)
    .def("shape", &UnifiedTensor::Shape)
    .def("device", &UnifiedTensor::Device)
    .def("size", &UnifiedTensor::Size, py::arg("dim"))
    .def("stride", &UnifiedTensor::Stride, py::arg("dim"))
    .def("numel", &UnifiedTensor::Numel);

  py::class_<CUDARandomSampler>(m, "CUDARandomSampler")
    .def(py::init<const Graph*>())
    .def("sample", &CUDARandomSampler::Sample,
         py::arg("ids"), py::arg("req_num"))
    .def("sample_with_edge", &CUDARandomSampler::SampleWithEdge,
         py::arg("ids"), py::arg("req_num"))
    .def("cal_nbr_prob", &CUDARandomSampler::CalNbrProb,
         py::arg("k"), py::arg("last_prob"), py::arg("nbr_last_prob"),
         py::arg("nbr_graph_"), py::arg("cur_prob"));

  py::class_<CUDARandomNegativeSampler>(m, "CUDARandomNegativeSampler")
    .def(py::init<const Graph*>())
    .def("sample", &CUDARandomNegativeSampler::Sample,
         py::arg("req_num"), py::arg("trials_num"), py::arg("padding"));

  py::class_<CUDAInducer>(m, "CUDAInducer")
    .def(py::init<int32_t>())
    .def("init_node", &CUDAInducer::InitNode,
         py::arg("seed"))
    .def("induce_next", &CUDAInducer::InduceNext,
         py::arg("srcs"), py::arg("nbrs"), py::arg("nbrs_num"));

  py::class_<CUDAHeteroInducer>(m, "CUDAHeteroInducer")
    .def(py::init<std::unordered_map<std::string, int32_t>>())
    .def("init_node", &CUDAHeteroInducer::InitNode,
         py::arg("seed"))
    .def("induce_next", &CUDAHeteroInducer::InduceNext,
         py::arg("hetero_nbrs"));

  py::class_<CUDASubGraphOp>(m, "CUDASubGraphOp")
    .def(py::init<const Graph*>())
    .def("node_subgraph", &CUDASubGraphOp::NodeSubGraph,
         py::arg("srcs"), py::arg("with_edge"));
#endif
}
