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
/*
#include <torch/extension.h>

#include "graphlearn_torch/include/grin/grin_graph.h"

#ifndef GRAPHLEARN_TORCH_INCLUDE_GRIN_SAMPLER_H_
#define GRAPHLEARN_TORCH_INCLUDE_GRIN_SAMPLER_H_

class GrinSampler {
public:
  GrinSampler(GrinGraph* graph) : graph_(graph) {}
  virtual ~GrinSampler() {}

  // return: (nbrs, nbrs_num)
  virtual std::tuple<torch::Tensor, torch::Tensor> Sample(
    const torch::Tensor& nodes, int32_t req_num) = 0;

protected:
  GrinGraph *graph_;
};

#endif  // GRAPHLEARN_TORCH_INCLUDE_GRIN_SAMPLER_H_
*/