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

#include "graphlearn_torch/include/grin/grin_utils.h"

GRIN_VERTEX_LIST GetVertexListByType(GRIN_GRAPH graph,
                                    GRIN_VERTEX_TYPE vtype) {
  auto vl = grin_get_vertex_list_by_type(graph, vtype);
  return vl;
}

GRIN_EDGE_LIST GetEdgeListByType(GRIN_GRAPH graph,
                                GRIN_EDGE_TYPE etype) {
  auto vl = grin_get_edge_list_by_type(graph, etype);
  return vl;
}