# Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from abc import ABC, abstractmethod
import torch
from ... import py_graphlearn_torch as pywrap

class GrinFeature(ABC):
  def __getitem__(self, ids: torch.Tensor) -> torch.Tensor:
    return self.cpu_get(ids)

  @abstractmethod
  def cpu_get(self, ids: torch.Tensor) -> torch.Tensor:
    pass

  @abstractmethod
  # @property
  def shape(self) -> torch.Tensor:
    pass

  @abstractmethod
  # @property
  def size(self, dim: int) -> torch.int64:
    pass

class GrinVertexFeature(GrinFeature):
  def __init__(self, uri, vertex_type, id2index=None):
    self._feature = pywrap.GrinVertexFeature(uri, vertex_type)
    self._id2index = id2index

  def cpu_get(self, ids: torch.Tensor) -> torch.Tensor:
    return self._feature.cpu_get(ids)
  
  def get_labels(self, ids: torch.Tensor) -> torch.Tensor:
    return self._feature.get_labels(ids)

  @property
  def shape(self) -> torch.Tensor:
    return torch.tensor(self._feature.shape(), dtype=torch.int64)

  @property
  def size(self, dim: int) -> torch.int64:
    return torch.tensor(self._feature.size(dim), dtype=torch.int64)


class GrinEdgeFeature(GrinFeature):
  def __init__(self, uri, edge_type, id2index=None):
    self._feature = pywrap.GrinEdgeFeature(uri, edge_type)
    self._id2index = id2index

  def cpu_get(self, ids: torch.Tensor) -> torch.Tensor:
    return self._feature.cpu_get(ids)

  @property
  def shape(self) -> torch.Tensor:
    return torch.tensor(self._feature.shape(), dtype=torch.int64)

  @property
  def size(self, dim: int) -> torch.int64:
    return torch.tensor(self._feature.size(dim), dtype=torch.int64)
