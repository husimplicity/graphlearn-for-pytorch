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

import logging
from multiprocessing.reduction import ForkingPickler
from typing import Dict, List, Optional, Union, Literal

import torch

from ...typing import NodeType, EdgeType, TensorDataType

from .grin_graph import GrinGraph
from .grin_feature import GrinFeature, GrinVertexFeature

class GrinDataset:
  def __init__(
    self,
    uri: str,
    graph: Union[GrinGraph, Dict[EdgeType, GrinGraph]] = None,
    node_features: Union[GrinFeature, Dict[NodeType, GrinFeature]] = None,
    edge_features: Union[GrinFeature, Dict[NodeType, GrinFeature]] = None,
    edge_dir = 'out',
  ):
    self.uri = uri
    self.graph = graph
    self.node_features = node_features
    self.edge_features = edge_features
    self.edge_dir = edge_dir

  def init_graph(self, edge_type_name: Union[str, List[str]]):
    # directly init from storage below grin
    # no need for topology
    if isinstance(edge_type_name, str):
      self.graph = GrinGraph(uri=self.uri, edge_type_name=edge_type_name)
    elif isinstance(edge_type_name, List):
      self.graph = {}
      for etype in edge_type_name:
        g = GrinGraph(uri=self.uri, edge_type_name=etype)
        edge_type = (g.src_type_name, etype, g.dst_type_name)
        self.graph[edge_type] = g

  def init_node_feat(
    self,
    num_props: int,
    id2idx: Dict[NodeType, TensorDataType]
  ):
    # id2idx provides external ids
    # vertex type must be provided to Grin for init
    self.node_features = {}
    for ntype, idx in id2idx.items():
      self.node_features[ntype] = GrinVertexFeature(
        uri=self.uri,
        vertex_type=ntype,
        num_props=num_props,
        id2index=idx
      )


  def get_graph(self, etype: Optional[EdgeType] = None):
    if isinstance(self.graph, GrinGraph):
      return self.graph
    if isinstance(self.graph, dict):
      assert etype is not None
      return self.graph.get(etype, None)
    return None

  def get_node_types(self):
    if isinstance(self.graph, dict):
      if not hasattr(self, '_node_types'):
        ntypes = set()
        for etype in self.graph.keys():
          ntypes.add(etype[0])
          ntypes.add(etype[2])
        self._node_types = list(ntypes)
      return self._node_types
    return None

  def get_edge_types(self):
    if isinstance(self.graph, dict):
      if not hasattr(self, '_edge_types'):
        self._edge_types = list(self.graph.keys())
      return self._edge_types
    return None

  def get_node_feature(self, ntype: Optional[str]=None):
    if ntype is None:
      return self.node_features
    return self.node_features[ntype]

  def get_edge_feature(self, etype: Optional[EdgeType]=None):
    if etype is None:
      return self.edge_features
    return self.edge_features[etype]

  def get_node_label(self, n_ids):
    if isinstance(n_ids, tuple):
      ntype, ids = n_ids
      return self.node_features[ntype].get_labels(ids)
    return self.node_features.get_labels(n_ids)