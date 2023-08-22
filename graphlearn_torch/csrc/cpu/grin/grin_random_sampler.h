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

#include "graphlearn_torch/include/grin/grin_sampler.h"


#ifndef GRAPHLEARN_TORCH_INCLUDE_GRIN_RANDOM_SAMPLER_H_
#define GRAPHLEARN_TORCH_INCLUDE_GRIN_RANDOM_SAMPLER_H_

class GrinRandomSampler : public GrinSampler {
public:
  GrinRandomSampler(GrinGraph* graph) : GrinSampler(graph) {}
  ~GrinRandomSampler() {}

  std::tuple<torch::Tensor, torch::Tensor> Sample(
    const torch::Tensor& nodes, int32_t req_num) override;

private:
  void FillNbrsNum(const int64_t* nodes,
                   const int32_t bs,
                   const int32_t req_num,
                   int64_t* out_nbr_num);

  void CSRRowWiseSample(const int64_t* nodes,
                        const int64_t* nbrs_offset,
                        const int32_t bs,
                        const int32_t req_num,
                        int64_t* out_nbrs);

};

#endif // GRAPHLEARN_TORCH_CPU_RANDOM_SAMPLER_H_
