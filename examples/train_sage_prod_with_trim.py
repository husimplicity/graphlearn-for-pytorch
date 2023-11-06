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

import os
import time
import torch

import numpy as np
import os.path as osp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from numpy import genfromtxt

from torch_geometric.nn import GraphSAGE
from ogb.nodeproppred import PygNodePropPredDataset
from tqdm import tqdm

import graphlearn_torch as glt


def run(rank, glt_ds, train_idx,
        num_features, num_classes, trimmed):

  train_loader = glt.loader.NeighborLoader(glt_ds,
                                           [15,10,5],
                                           train_idx,
                                           batch_size=1024,
                                           shuffle=True,
                                           device=torch.device('cpu'))
  print(f'Rank {rank} build graphlearn_torch NeighborLoader Done.')
  model = GraphSAGE(
    in_channels=num_features,
    hidden_channels=256,
    num_layers=3,
    out_channels=num_classes,
  ).to(torch.device('cpu'))

  optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

  for epoch in range(10):
    # model.train()
    start = time.time()
    total_examples = total_loss = 0
    sizes = []
    for batch in tqdm(train_loader):
      print(batch.edge_index.size(1))
      sample_time = time.time()
      print(f'sample time: {sample_time - start}')
      optimizer.zero_grad()
      # if trimmed:
      #   out = model(
      #     batch.x, batch.edge_index,
      #     num_sampled_nodes_per_hop=batch.num_sampled_nodes,
      #     num_sampled_edges_per_hop=batch.num_sampled_edges,
      #   )[:batch.batch_size].log_softmax(dim=-1)
      # else:
      out = model(
        batch.x, batch.edge_index
      )[:batch.batch_size].log_softmax(dim=-1)
      loss = F.nll_loss(out, batch.y[:batch.batch_size])
      loss.backward()
      optimizer.step()
      total_examples += batch.batch_size
      total_loss += float(loss) * batch.batch_size

      start = time.time()
      print(f'training time: {start - sample_time}')

      # print(#f'Epoch: {epoch:03d}, Loss: {(total_loss / total_examples):.4f},',
      #       f'Epoch Time: {end - start}')


if __name__ == '__main__':
  world_size = torch.cuda.device_count()
  start = time.time()
  root = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ogbn-products')
  dataset = PygNodePropPredDataset('ogbn-products', root)
  print(dataset.meta_info['add_inverse_edge'])
  split_idx = dataset.get_idx_split()
  data = dataset[0]
  train_idx = split_idx['train']
  # torch.save(train_idx, 'train_idx.pt')
  # test_idx = split_idx['test']
  # torch.save(test_idx, 'test_idx.pt')
  # train_idx = torch.arange(244902)
  print(f'Load data cost {time.time()-start} s.')

  start = time.time()
  print('Build graphlearn_torch dataset...')
  glt_dataset = glt.data.Dataset()
  glt_dataset.init_graph(
    edge_index=data.edge_index,
    graph_mode='CPU',
    directed=True
  )
  print(glt_dataset.get_graph().edge_count)
  glt_dataset.init_node_features(
    node_feature_data=data.x,
    sort_func=glt.data.sort_by_in_degree,
    with_gpu=False
    # split_ratio=1,
    # device_group_list=[glt.data.DeviceGroup(0, [0])],
  )
  glt_dataset.init_node_labels(node_label_data=data.y)
  print(f'Build graphlearn_torch csr_topo and feature cost {time.time() - start} s.')

  run(0, glt_dataset, train_idx, 100, 47, False)