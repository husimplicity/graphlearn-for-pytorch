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
#define GRIN_ENABLE_EDGE_LIST
#define GRIN_ENABLE_EDGE_LIST_ARRAY
#define GRIN_ENABLE_ADJACENT_LIST
#define GRIN_ASSUME_ALL_VERTEX_LIST_SORTED
#define GRIN_ENABLE_ADJACENT_LIST_ARRAY
#define GRIN_ENABLE_VERTEX_EXTERNAL_ID_OF_INT64

#include "graphlearn_torch/grin/grin_feature.h"

#include "grin/include/index/external_id.h"
#include "grin/include/property/property.h"
#include "grin/include/property/propertylist.h"
#include "grin/include/property/value.h"


torch::Tensor GrinVertexFeature::cpu_get(const torch::Tensor& ex_ids) { 
  int64_t bs = ex_ids.size(0);
  int64_t* ex_ids_ptr = ex_ids.data_ptr<int64_t>();
  auto prop = grin_get_vertex_property_by_name(graph_, vertex_type_, "features");
  // auto props = grin_get_vertex_property_list_by_type(graph_, vertex_type_);
  // auto prop = grin_get_vertex_property_from_list(graph_, props, 0);
  prop = 2;
  auto v0 = grin_get_vertex_by_external_id_of_int64(graph_, ex_ids_ptr[0]);
  // std::cout << "v0:" << v0 << std::endl;
  size_t* num_props = new size_t;
  grin_get_vertex_property_value_of_float_array(graph_, v0, prop, num_props);
  size_t np = *num_props;
  // std::cout << "prop feat: " << prop << " np: " << np << std::endl;
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  std::vector<torch::Tensor> vfeats;
  vfeats.resize(bs);
  at::parallel_for(0, bs, 1, [&](int32_t start, int32_t end) {
    for (int32_t i = start; i < end; i++) {
      auto v = grin_get_vertex_by_external_id_of_int64(graph_, ex_ids_ptr[i]);
      float* p = const_cast<float*>(
          grin_get_vertex_property_value_of_float_array(graph_, v, prop, num_props));
      vfeats[i] = torch::from_blob(p, {np}, options);
      grin_destroy_vertex(graph_, v);
    }
    grin_destroy_vertex_property(graph_, prop);
  });

  delete num_props;
  grin_destroy_vertex_property(graph_, prop);
  auto vf = torch::stack(vfeats, 0);
  return vf;
}

torch::Tensor GrinVertexFeature::get_labels(const torch::Tensor& ex_ids) {
  int64_t bs = ex_ids.size(0);
  int64_t* ex_ids_ptr = ex_ids.data_ptr<int64_t>();
  auto prop = grin_get_vertex_property_by_name(graph_, vertex_type_, "label");
  prop = 0;
  torch::Tensor vlabels = torch::empty({bs}, torch::kInt64);

  at::parallel_for(0, bs, 1, [&](int32_t start, int32_t end) {
    for (int32_t i = start; i < end; ++i) {
      auto v = grin_get_vertex_by_external_id_of_int64(graph_, ex_ids_ptr[i]);
      int64_t vlabel = grin_get_vertex_property_value_of_int32(graph_, v, prop);
      grin_destroy_vertex(graph_, v);
      vlabels[i] = vlabel;
    }
  });
  grin_destroy_vertex_property(graph_, prop);
  return vlabels;
}

// torch::Tensor GrinEdgeFeature::cpu_get(const torch::Tensor& ex_ids) {
//   int64_t bs = ex_ids.size(0);
//   auto props = grin_get_edge_property_list_by_type(graph_, edge_type_);
//   int64_t num_props = grin_get_edge_property_list_size(graph_, props);
//   torch::Tensor efeats = torch::empty({bs, num_props}, torch::kFloat32);

// // TODO: Currently edge_id not supported by grin
// //   at::parallel_for(0, num_props, 1, [&](int32_t start, int32_t end) {
// //     for (size_t j = start; j < end; j++) {
// //       auto prop = grin_get_edge_property_from_list(graph_, props, j);
// //       for (int32_t i = 0; i < bs; i++) {
// //         auto e = grin_get_edge_by_external_id_of_int64(graph_, ex_ids[i]);
// //         efeats[i][j] = grin_get_edge_property_value_of_float(graph_, e, prop);
// //         grin_destroy_edge(graph_, e);
// //       }
// //       grin_destroy_edge_property(graph_, prop);
// //     }
// //   });
//   grin_destroy_edge_property_list(graph_, props);
//   return efeats;
// }
