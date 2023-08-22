import unittest
import torch

from graphlearn_torch.data.grin import GrinGraph, GrinVertexFeature
from graphlearn_torch.sampler import NeighborSampler
from graphlearn_torch.utils import tensor_equal_with_device

g = GrinGraph(uri="gart://127.0.0.1:23760?read_epoch=80&total_partition_num=1&local_partition_num=1&start_partition_id=0&meta_prefix=gart_meta_",
              edge_type_name="paper_cites_paper")
print(g.src_type_name)
f = GrinVertexFeature(uri="gart://127.0.0.1:23760?read_epoch=80&total_partition_num=1&local_partition_num=1&start_partition_id=0&meta_prefix=gart_meta_",
                      vertex_type="paper")
print(f.cpu_get(torch.tensor([0,2]), 4))

sampler = NeighborSampler(g, [2], device=torch.device('cpu'))
sampler_out = sampler.sample_from_nodes(torch.tensor([0, 2]))
print(sampler_out)