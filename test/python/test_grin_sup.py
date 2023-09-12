import time, tqdm
import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE
from ogb.nodeproppred import Evaluator

from graphlearn_torch.data.grin import GrinDataset
from graphlearn_torch.loader import NeighborLoader, NeighborLoader


@torch.no_grad()
def test(model, test_loader, dataset_name):
    evaluator = Evaluator(name=dataset_name)
    model.eval()
    xs = []
    y_true = []
    for batch in tqdm.tqdm(test_loader):
        x = model(batch.x, batch.edge_index)[:batch.batch_size]
        xs.append(x.cpu())
        y_true.append(batch.y[:batch.batch_size].clone().cpu())

    xs = [t.to(device) for t in xs]
    y_true = [t.to(device) for t in y_true]
    y_pred = torch.cat(xs, dim=0).argmax(dim=-1, keepdim=True)
    y_true = torch.cat(y_true, dim=0).unsqueeze(-1)
    test_acc = evaluator.eval({
        'y_true': y_true,
        'y_pred': y_pred,
    })['acc']
    return test_acc

grindataset = GrinDataset(uri="gart://127.0.0.1:23760?read_epoch=6429&total_partition_num=1&local_partition_num=1&start_partition_id=0&meta_prefix=gart_meta_")

grindataset.init_graph(edge_type_name="paper_cites_paper")
grindataset.init_node_feat(
    num_props=100, id2idx={"paper": torch.arange(0, 2449029, dtype=torch.int64)})
grindataset.init_node_label(n_ids=torch.arange(0, 2449029, dtype=torch.int64))

seeds = torch.randperm(2449029, dtype=torch.int64)
train_ids = seeds[:round(2449029 * 0.1)]
test_ids = seeds[round(2449029 * 0.1):]
device = torch.device('cpu')

train_loader = NeighborLoader(grindataset,
                              [15,10,5],
                              input_nodes=train_ids,
                              batch_size=1024,
                              shuffle=True,
                              device=device,
)


test_loader = NeighborLoader(grindataset,
                             [15,10,5],
                             input_nodes=test_ids,
                             batch_size=4096,
                             shuffle=False,
                             device=device,
)

model = GraphSAGE(
    in_channels=100,
    hidden_channels=256,
    num_layers=3,
    out_channels=47,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

for epoch in range(20):
    model.train()
    start = time.time()
    total_examples = total_loss = 0
    for batch in tqdm.tqdm(train_loader):
        optimizer.zero_grad()

        out = model(
            batch.x, batch.edge_index
        )[:batch.batch_size].log_softmax(dim=-1)

        loss = F.nll_loss(out, batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()
        total_examples += batch.batch_size
        total_loss += float(loss) * batch.batch_size
        end = time.time()

    print(f'Epoch: {epoch:03d}, Loss: {(total_loss / total_examples):.4f},',
          f'Epoch Time: {end - start}')

    test_acc = test(model, test_loader, 'ogbn-products')
    print(f'Test Acc: {test_acc:.4f}\n')