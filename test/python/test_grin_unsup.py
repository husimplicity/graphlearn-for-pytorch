import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE
from sklearn.metrics import roc_auc_score, recall_score

from graphlearn_torch.data.grin import GrinGraph, GrinVertexFeature, GrinDataset
from graphlearn_torch.sampler import NeighborSampler, NegativeSampling
from graphlearn_torch.loader import NeighborLoader, LinkNeighborLoader
from graphlearn_torch.utils import tensor_equal_with_device

device = torch.device("cpu")
model = GraphSAGE(
    in_channels=18,
    hidden_channels=256,
    num_layers=3,
    out_channels=64,
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-7)

df = pd.read_csv("/root/wanglei/cora/cora.content", delimiter='\t', header=None)
edge_df = pd.read_csv("/root/wanglei/cora/cora.cites", delimiter='\t', header=None)

id2idx = {"paper": torch.from_numpy(df.iloc[:,0].to_numpy())}
edge_df = torch.from_numpy(edge_df.iloc[:,[0,1]].to_numpy().T)
edge_df_train = edge_df[:,:round(edge_df.size(1) * 0.8)]
edge_df_test = edge_df[:,round(edge_df.size(1) * 0.8):]
print(edge_df_train, edge_df_test)
grindataset = GrinDataset(uri="gart://127.0.0.1:23760?read_epoch=6429&total_partition_num=1&local_partition_num=1&start_partition_id=0&meta_prefix=gart_meta_")

grindataset.init_graph(edge_type_name="paper_cites_paper")
grindataset.init_node_feat(num_props=18, id2idx=id2idx)

neg_config = NegativeSampling(mode="binary", seeds=id2idx['paper'])
train_loader = LinkNeighborLoader(grindataset,
                            [15,10,5],
                            edge_label_index=edge_df_train,
                            batch_size=10,
                            neg_sampling=neg_config,
                            device=torch.device('cpu'),
                            )

test_loader = LinkNeighborLoader(grindataset,
                            [15,10,5],
                            edge_label_index=edge_df_test,
                            batch_size=10,
                            neg_sampling=neg_config,
                            device=torch.device('cpu'),
                            )



for epoch in range(50):
    model.train()
    total_loss = total_examples = 0
    for batch in train_loader:
        if batch.edge_index.shape[1] != 0:
            optimizer.zero_grad()
            h = model(batch.x, batch.edge_index)
            h_src = h[batch.edge_label_index[0]]
            h_dst = h[batch.edge_label_index[1]]
            link_pred = (h_src * h_dst).sum(dim=-1)  # Inner product.
            # print(link_pred)
            edge_label = batch.edge_label
            
            loss = F.binary_cross_entropy_with_logits(link_pred, edge_label)
            loss.backward()
            optimizer.step()

            total_loss += float(loss) * link_pred.numel()
            total_examples += link_pred.numel()
    print(total_loss)

    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for batch in train_loader:
            if batch.edge_index.shape[1] != 0:
                h = model(batch.x, batch.edge_index)
                h_src = h[batch.edge_label_index[0]]
                h_dst = h[batch.edge_label_index[1]]
                link_pred = (h_src * h_dst).sum(dim=-1).sigmoid().round()  # Inner product.
                edge_label = batch.edge_label

                preds.append(link_pred)
                targets.append(edge_label)

        pred = torch.cat(preds, dim=0).numpy()
        target = torch.cat(targets, dim=0).numpy()
        print(roc_auc_score(target, pred))