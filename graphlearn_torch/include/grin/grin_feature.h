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

#include <torch/extension.h>

#define GRIN_ENABLE_VERTEX_LIST_ARRAY
#define GRIN_ENABLE_EDGE_LIST
#define GRIN_ENABLE_EDGE_LIST_ARRAY

#include "predefine.h"

#include "grin/include/topology/vertexlist.h"
#include "grin/include/topology/edgelist.h"
#include "grin/include/topology/adjacentlist.h"
#include "grin/include/topology/structure.h"
#include "grin/include/partition/partition.h"
#include "grin/include/partition/topology.h"
#include "grin/include/property/type.h"
#include "grin/include/property/topology.h"


#ifndef GRAPHLEARN_TORCH_INCLUDE_GRIN_FEATURE_H_
#define GRAPHLEARN_TORCH_INCLUDE_GRIN_FEATURE_H_


class GrinFeature {
public:
  GrinFeature(const char* uri,
              const std::string& type_name) {
    partitioned_graph_ = grin_get_partitioned_graph_from_storage(uri);
    local_partitions_ = grin_get_local_partition_list(partitioned_graph_);
    partition_ = grin_get_partition_from_list(
      partitioned_graph_, local_partitions_, 0);
    graph_ = grin_get_local_graph_by_partition(partitioned_graph_, partition_);
  }
  ~GrinFeature() {}

  virtual torch::Tensor cpu_get(const torch::Tensor& ex_ids) = 0;

protected:
  GRIN_PARTITIONED_GRAPH          partitioned_graph_;
  GRIN_PARTITION_LIST             local_partitions_;
  GRIN_PARTITION                  partition_;
  GRIN_GRAPH                      graph_;

};

class GrinVertexFeature : public GrinFeature {
public:
  GrinVertexFeature(const char* uri,
                    const std::string& type_name)
    : GrinFeature(uri, type_name) {
    vertex_type_ = grin_get_vertex_type_by_name(graph_, type_name.c_str());
    vertex_list_ = grin_get_vertex_list_by_type(graph_, vertex_type_);
    num_vertices_ = grin_get_vertex_list_size(graph_, vertex_list_);
  }

  ~GrinVertexFeature() {
    grin_destroy_vertex_list(graph_, vertex_list_);
  }

  torch::Tensor cpu_get(const torch::Tensor& ex_ids) override;
  torch::Tensor get_labels(const torch::Tensor& ex_ids);

private:
  GRIN_VERTEX_TYPE  vertex_type_;
  GRIN_VERTEX_LIST  vertex_list_;
  size_t            num_vertices_;
};


class GrinEdgeFeature : public GrinFeature {
public:
  GrinEdgeFeature(const char* uri,
                  const std::string& type_name)
    : GrinFeature(uri, type_name) {
    edge_type_ = grin_get_edge_type_by_name(graph_, type_name.c_str());
    edge_list_ = grin_get_edge_list_by_type(graph_, edge_type_);
    num_edges_ = grin_get_edge_list_size(graph_, edge_list_);
  }

  ~GrinEdgeFeature() {
    grin_destroy_edge_list(graph_, edge_list_);
  }

  torch::Tensor cpu_get(const torch::Tensor& ex_ids) override;

private:
  GRIN_EDGE_TYPE  edge_type_;
  GRIN_EDGE_LIST  edge_list_;
  size_t          num_edges_;
};


#endif  // GRAPHLEARN_TORCH_INCLUDE_GRIN_FEATURE_H_
