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

from .. import py_graphlearn_torch as pywrap

class GrinGraph:
  def __init__(self, uri: str, edge_type_name: str):
    self._graph = pywrap.GrinGraph(uri, edge_type_name)

  @property
  def graph_handler(self):
    return self._graph

  @property
  def row_count(self):
    return self._graph.col_count()
  
  @property
  def src_type_name(self):
    return self._graph.get_src_type_name()
  
  @property
  def dst_type_name(self):
    return self._graph.get_dst_type_name()

  @property
  def edge_type_name(self):
    return self._graph.get_edge_type_name()
