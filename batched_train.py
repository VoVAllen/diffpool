from __future__ import division
from __future__ import print_function

import argparse

import torch
import torch.optim as optim
from tqdm import tqdm
import config
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, random_split

from batched_model import BatchedModel
from dataset import TUDataset, CollateFn


def main():
    config.writer = SummaryWriter()

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--epochs', type=int, default=700,
                        help='Number of epochs to train.')
    parser.add_argument('--link-pred', action='store_true', default=False,
                        help='Enable Link Prediction Loss')
    parser.add_argument('--dataset', default='ENZYMES', help="Choose dataset: ENZYMES, DD")

    args = parser.parse_args()
    config.writer.add_text("args", str(args))
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"

    dataset = TUDataset(args.dataset)
    max_num_nodes = max([item[0].shape[0] for item in dataset])
    train_data, test_data = random_split(dataset, (540, 60))
    train_loader = DataLoader(train_data, batch_size=20, shuffle=True, collate_fn=CollateFn(device))
    test_loader = DataLoader(test_data, batch_size=20, shuffle=True, collate_fn=CollateFn(device))

    model = BatchedModel(pool_size=int(max_num_nodes * 0.25), device=device).to(device)
    model.train()
    optimizer = optim.Adam(model.parameters())

    for e in tqdm(range(args.epochs)):
        config.e = e
        epoch_losses_list = []
        true_sample = 0
        for i, (adj, features, masks, batch_labels) in enumerate(train_loader):
            graph_feat = model(features, adj, masks)
            output = model.classifier(graph_feat)
            loss = model.loss(output, batch_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            optimizer.zero_grad()

            epoch_losses_list.append(loss.item())
            true_sample += (output.argmax(dim=1).long() == batch_labels.long()).float().sum().item()

        acc = true_sample / 540
        config.writer.add_scalar("Epoch Acc", acc, e)
        tqdm.write(f"Epoch:{e}  \t train_acc:{acc:.2f}")

        test_loss_list = []
        true_sample = 0
        with torch.no_grad():
            for i, (adj, features, masks, batch_labels) in enumerate(test_loader):
                graph_feat = model(features, adj, masks)
                output = model.classifier(graph_feat)
                loss = model.loss(output, batch_labels)
                test_loss_list.append(loss.item())
                true_sample += (
                            output.argmax(dim=1).long() == batch_labels.long()).float().sum().item()
        acc = true_sample / 60
        config.writer.add_scalar("Epoch Acc", acc, e)
        tqdm.write(f"Epoch:{e}  \t val_acc:{acc:.2f}")


main()
