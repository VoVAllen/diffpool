from __future__ import division
from __future__ import print_function

import argparse

import torch
import torch.optim as optim
from torchvision.datasets import MNIST
from tqdm import tqdm
import utils
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, random_split

from MNIST import GraphTransform
from batched_model import BatchedModel
from dataset import TUDataset, CollateFn


def main():
    utils.writer = SummaryWriter()

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--epochs', type=int, default=700,
                        help='Number of epochs to train.')
    parser.add_argument('--link-pred', action='store_true', default=False,
                        help='Enable Link Prediction Loss')
    parser.add_argument('--dataset', default='ENZYMES', help="Choose dataset: ENZYMES, DD")
    parser.add_argument('--batch-size', default=256, type=int, help="Choose dataset: ENZYMES, DD")
    parser.add_argument('--train-ratio', default=0.9, type=float, help="Train/Val split ratio")
    parser.add_argument('--pool-ratio', default=0.25, type=float, help="Train/Val split ratio")

    args = parser.parse_args()
    utils.writer.add_text("args", str(args))
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"

    # dataset = TUDataset(args.dataset)
    dataset = MNIST(root="~/.torch/data/", transform=GraphTransform(device), download=True)
    dataset_size = len(dataset)
    train_size = int(dataset_size * args.train_ratio)
    test_size = dataset_size - train_size
    max_num_nodes = max([item[0][0].shape[0] for item in dataset])
    n_classes = int(max([item[1] for item in dataset])) + 1
    train_data, test_data = random_split(dataset, (train_size, test_size))
    input_shape = int(dataset[0][0][1].shape[-1])
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              collate_fn=CollateFn(device))
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True,
                             collate_fn=CollateFn(device))

    model = BatchedModel(pool_size=int(max_num_nodes * args.pool_ratio), device=device,
                         link_pred=args.link_pred, input_shape=input_shape, n_classes=n_classes).to(
        device)
    model.train()
    optimizer = optim.Adam(model.parameters())

    for e in tqdm(range(args.epochs)):
        utils.e = e
        epoch_losses_list = []
        true_sample = 0
        model.train()
        for i, (adj, features, masks, batch_labels) in enumerate(train_loader):
            utils.train_iter += 1
            graph_feat = model(features, adj, masks)
            output = model.classifier(graph_feat)
            loss = model.loss(output, batch_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            optimizer.zero_grad()

            epoch_losses_list.append(loss.item())
            iter_true_sample = (output.argmax(dim=1).long() == batch_labels.long()). \
                float().sum().item()
            iter_acc = float(iter_true_sample) / output.shape[0]
            utils.writer.add_scalar("iter train acc", iter_acc, utils.train_iter)
            print(f"{utils.train_iter} iter train acc: {iter_acc}")
            true_sample += iter_true_sample

        acc = true_sample / train_size
        utils.writer.add_scalar("Epoch Acc", acc, e)
        tqdm.write(f"Epoch:{e}  \t train_acc:{acc:.2f}")

        test_loss_list = []
        true_sample = 0
        model.eval()
        with torch.no_grad():
            for i, (adj, features, masks, batch_labels) in enumerate(test_loader):
                utils.test_iter += 1
                graph_feat = model(features, adj, masks)
                output = model.classifier(graph_feat)
                loss = model.loss(output, batch_labels)
                test_loss_list.append(loss.item())
                iter_true_sample = (output.argmax(dim=1).long() == batch_labels.long()). \
                    float().sum().item()
                iter_acc = float(iter_true_sample) / output.shape[0]
                utils.writer.add_scalar("iter test acc", iter_acc, utils.test_iter)
                print(f"{utils.test_iter} iter test acc: {iter_acc}")
                true_sample += iter_true_sample
        acc = true_sample / test_size
        utils.writer.add_scalar("Epoch Acc", acc, e)
        tqdm.write(f"Epoch:{e}  \t val_acc:{acc:.2f}")

        # Visualize
        # adj, features, masks, batch_labels = test_loalder
        # model()


main()
