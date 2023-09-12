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

#define GRIN_ENABLE_VERTEX_LIST_ARRAY
#define GRIN_ENABLE_ADJACENT_LIST_ITERATOR
#define GRIN_ENABLE_ADJACENT_LIST
#define GRIN_ASSUME_ALL_VERTEX_LIST_SORTED
#define GRIN_ENABLE_VERTEX_EXTERNAL_ID_OF_INT64

#include "graphlearn_torch/csrc/cpu/grin/grin_random_sampler.h"
#include "grin/extension/include/indexed_adjacent_list.h"

#include "grin/include/index/order.h"
// #include "grin/include/topology/adjacentlist.h"


#include <cstdint>
#include <random>

std::tuple<torch::Tensor, torch::Tensor> 
GrinRandomSampler::Sample(const torch::Tensor& nodes, int32_t req_num) {
  if (req_num < 0) req_num = std::numeric_limits<int32_t>::max();
  const int64_t* nodes_ptr = nodes.data_ptr<int64_t>();
  int64_t bs = nodes.size(0);
  torch::Tensor nbrs_num = torch::empty(bs, nodes.options());
  auto nbrs_num_ptr = nbrs_num.data_ptr<int64_t>();
  FillNbrsNum(nodes_ptr, bs, req_num, nbrs_num_ptr);
  int64_t nbrs_offset[bs + 1];
  nbrs_offset[0] = 0;
  for(int64_t i = 1; i <= bs; ++i) {
    nbrs_offset[i] = nbrs_offset[i - 1] + nbrs_num_ptr[i - 1];
  }

  torch::Tensor nbrs = torch::empty(nbrs_offset[bs], nodes.options());
  CSRRowWiseSample(
    nodes_ptr, nbrs_offset, bs, req_num, nbrs.data_ptr<int64_t>());
  return std::make_tuple(nbrs, nbrs_num);
}

void GrinRandomSampler::FillNbrsNum(const int64_t* nodes,
                                    const int32_t bs,
                                    const int32_t req_num,
                                    int64_t* out_nbr_num) {
  GRIN_GRAPH graph = graph_->GetGraph();
  GRIN_EDGE_TYPE etype = graph_->GetEdgeType();
  at::parallel_for(0, bs, 1, [&](int32_t start, int32_t end) {
    for(int32_t i = start; i < end; i++){
      auto v = nodes[i];
      auto src = grin_get_vertex_by_external_id_of_int64(graph, v);
      if (src != GRIN_NULL_VERTEX) {
        auto src_adj_list = grin_get_adjacent_list_by_edge_type(
          graph, GRIN_DIRECTION::OUT, src, etype);
        auto src_idx_adj_list = grin_get_indexed_adjacent_list(graph, src_adj_list);
        auto src_degree = grin_get_indexed_adjacent_list_size(graph, src_idx_adj_list);

        out_nbr_num[i] = req_num < src_degree ? (int64_t)req_num : src_degree;
        grin_destroy_indexed_adjacent_list(graph, src_idx_adj_list);
        grin_destroy_adjacent_list(graph, src_adj_list);
        grin_destroy_vertex(graph, src);
      } else {
        out_nbr_num[i] = 0;
      }
    }
  });
  grin_destroy_edge_type(graph, etype);
  // grin_destroy_graph(graph);
}

void GrinRandomSampler::CSRRowWiseSample(
    const int64_t* nodes,
    const int64_t* nbrs_offset,
    const int32_t bs,
    const int32_t req_num,
    int64_t* out_nbrs) {
  GRIN_GRAPH graph = graph_->GetGraph();
  GRIN_EDGE_TYPE etype = graph_->GetEdgeType();
  at::parallel_for(0, bs, 1, [&](int32_t start, int32_t end) {
    for(int32_t i = start; i < end; ++i) {
      auto v = nodes[i];
      auto src = grin_get_vertex_by_external_id_of_int64(graph, v);
      // std::cout << "v: " << v << " src: " << src << std::endl;
      if (src != GRIN_NULL_VERTEX) {
        auto src_adj_list = grin_get_adjacent_list_by_edge_type(
          graph, GRIN_DIRECTION::OUT, src, etype);
        auto src_idx_adj_list = grin_get_indexed_adjacent_list(graph, src_adj_list);
        auto src_degree = grin_get_indexed_adjacent_list_size(graph, src_idx_adj_list);

        if (req_num < src_degree) {
          thread_local static std::random_device rd;
          thread_local static std::mt19937 engine(rd());
          std::uniform_int_distribution<> dist(0, req_num - 1);
          for (int32_t j = 0; j < req_num; ++j) {
            auto out_v = grin_get_neighbor_from_indexed_adjacent_list(
              graph, src_idx_adj_list, dist(engine));
            auto out_id = grin_get_vertex_external_id_of_int64(graph, out_v);
            out_nbrs[nbrs_offset[i] + j] = out_id;
            grin_destroy_vertex(graph, out_v);
          }
        } else {
          for (int32_t j = 0; j < src_degree; ++j) {
            auto out_v = grin_get_neighbor_from_indexed_adjacent_list(
              graph, src_idx_adj_list, j);
            auto out_id = grin_get_vertex_external_id_of_int64(graph, out_v);
            out_nbrs[nbrs_offset[i] + j] = out_id;
            grin_destroy_vertex(graph, out_v);
          }
        }
        grin_destroy_indexed_adjacent_list(graph, src_idx_adj_list);
        grin_destroy_adjacent_list(graph, src_adj_list);
        grin_destroy_vertex(graph, src);
      }
    }
  });
  // std::cout << "Here" << std::endl;
  grin_destroy_edge_type(graph, etype);
  // grin_destroy_graph(graph);
} 
