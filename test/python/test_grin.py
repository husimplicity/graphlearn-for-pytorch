import pandas as pd
import torch

from graphlearn_torch.data.grin import GrinGraph, GrinVertexFeature, GrinDataset
from graphlearn_torch.sampler import NeighborSampler
from graphlearn_torch.loader import NeighborLoader
from graphlearn_torch.utils import tensor_equal_with_device

# g = GrinGraph(uri="gart://127.0.0.1:23760?read_epoch=7&total_partition_num=1&local_partition_num=1&start_partition_id=0&meta_prefix=gart_meta_",
#               edge_type_name="paper_cites_paper")
# print(g.src_type_name)
# print(g.dst_type_name)
# f = GrinVertexFeature(uri="gart://127.0.0.1:23760?read_epoch=7&total_partition_num=1&local_partition_num=1&start_partition_id=0&meta_prefix=gart_meta_",
#                       vertex_type="paper", num_props=18)
# print(f[torch.tensor([35])])

# sampler = NeighborSampler(g, [3, 1], device=torch.device('cpu'))
# sampler_out = sampler.sample_from_nodes(torch.tensor([35]))
# print(sampler_out)
df = pd.read_csv("/root/wanglei/cora/cora.content", delimiter='\t', header=None)
id2idx = {"paper": torch.from_numpy(df.iloc[:,0].to_numpy())}
labels = {"paper": torch.from_numpy(df.iloc[:,1].to_numpy())}
grindataset = GrinDataset(uri="gart://127.0.0.1:23760?read_epoch=7&total_partition_num=1&local_partition_num=1&start_partition_id=0&meta_prefix=gart_meta_")

grindataset.init_graph(edge_type_name=["paper_cites_paper"])
grindataset.init_node_feat(num_props=18, id2idx=id2idx)


print(grindataset.get_node_feature("paper")[torch.tensor([35, 31336])])

loader = NeighborLoader(grindataset,
                        [2,1],
                        ('paper', id2idx["paper"]),
                        batch_size=5,
                        device=torch.device('cpu'))

for batch in loader:
    print(batch)