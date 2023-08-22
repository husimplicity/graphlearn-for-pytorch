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
#define GRIN_ENABLE_ADJACENT_LIST
#define GRIN_ASSUME_ALL_VERTEX_LIST_SORTED
#define GRIN_ENABLE_VERTEX_EXTERNAL_ID_OF_INT64

#include "predefine.h"

#include "grin/include/topology/vertexlist.h"
#include "grin/include/topology/edgelist.h"
#include "grin/include/topology/adjacentlist.h"
#include "grin/include/topology/structure.h"
#include "grin/include/partition/partition.h"
#include "grin/include/partition/topology.h"
#include "grin/include/property/type.h"
#include "grin/include/property/topology.h"
#include "grin/include/index/external_id.h"

#ifndef GRAPHLEARN_TORCH_INCLUDE_GRIN_GRAPH_H_
#define GRAPHLEARN_TORCH_INCLUDE_GRIN_GRAPH_H_

class GrinGraph {
public:
  GrinGraph(const char* uri,
            const std::string& edge_type_name) {
    partitioned_graph_ = grin_get_partitioned_graph_from_storage(uri);
    local_partitions_ = grin_get_local_partition_list(partitioned_graph_);
    partition_ = grin_get_partition_from_list(
      partitioned_graph_, local_partitions_, 0);
    std::cout << "partition: " << partition_ << std::endl;
    graph_ = grin_get_local_graph_by_partition(partitioned_graph_, partition_);
    edge_type_ = grin_get_edge_type_by_name(graph_, edge_type_name.c_str());
    auto src_types = grin_get_src_types_by_edge_type(graph_, edge_type_);
    src_type_ = grin_get_vertex_type_from_list(graph_, src_types, 0);
    auto dst_types = grin_get_dst_types_by_edge_type(graph_, edge_type_);
    dst_type_ = grin_get_vertex_type_from_list(graph_, dst_types, 0);
  };

  ~GrinGraph() {
    grin_destroy_graph(graph_);
    grin_destroy_partition_list(partitioned_graph_, local_partitions_);
    grin_destroy_partitioned_graph(partitioned_graph_);
  };

  GRIN_GRAPH GetGraph() {
    return graph_;
  }

  GRIN_EDGE_TYPE GetEdgeType() {
    return edge_type_;
  }

  GRIN_VERTEX_LIST GetRowPtr() {
    // return grin_get_vertex_list_by_type(graph_, src_type_);
    return NULL;
  }

  int64_t GetRowCount() {
    // return grin_get_vertex_list_size(graph_, GetRowPtr());
    return 0;
  }

  const char* GetSrcTypeName() {
    return grin_get_vertex_type_name(graph_, src_type_);
  }

  const char* GetDstTypeName() {
    return grin_get_vertex_type_name(graph_, dst_type_);
  }

  const char* GetEdgeTypeName() {
    return grin_get_edge_type_name(graph_, edge_type_);
  }

  int64_t GetEdgeCount() const {
    // TODO(wanglei)
    return 0;
  }


  int64_t GetColCount() const {
    return 0;
  }

private:
  friend class GrinRandomSampler;
  GRIN_PARTITIONED_GRAPH  partitioned_graph_;
  GRIN_PARTITION_LIST     local_partitions_;
  GRIN_PARTITION          partition_;
  GRIN_GRAPH              graph_;

  GRIN_EDGE_TYPE          edge_type_;
  GRIN_VERTEX_TYPE        src_type_;
  GRIN_VERTEX_TYPE        dst_type_;
  int64_t                 row_count_;
  int64_t                 edge_count_;
  int64_t                 col_count_;
};

#endif  // GRAPHLEARN_TORCH_INCLUDE_GRIN_GRAPH_H_
