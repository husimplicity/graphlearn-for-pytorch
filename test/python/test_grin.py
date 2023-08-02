import unittest
import torch

from graphlearn_torch.data.grin import GrinGraph
from graphlearn_torch.utils import tensor_equal_with_device

g = GrinGraph(uri="gart://127.0.0.1:23760?read_epoch=96&total_partition_num=1&local_partition_num=1&start_partition_id=0&meta_prefix=gart_meta_",
              edge_type_name="knows")
print(g.src_type_name)