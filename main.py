import argparse

import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader

from ptsne import train_parametric_tsne_model
from datasets import load_mnist_some_classes
from config import config

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--perplexity', '-p', type=int, default=None,
                    help='perplexity to use instead of one in the config')
parser.add_argument('--epochs', '-e', type=int, default=None,
                    help='perplexity to use instead of one in the config')
parser.add_argument('--batchsize', '-b', type=int, default=None,
                    help='batch size to use instead of one in the config')

args = parser.parse_args()


def get_batch_embeddings(model: torch.nn.Module,
                         input_points: Dataset,
                         batch_size: int,
                         ):
    """
    Yields final embeddings for every batch in dataset
    :param model:
    :param input_points:
    :param batch_size:
    :return:
    """
    model.eval()
    test_dl = DataLoader(input_points, batch_size=batch_size, shuffle=False)
    for batch_points, batch_labels in test_dl:
        with torch.no_grad():
            embeddings = model(batch_points)
            yield embeddings, batch_labels


def plot_embs(trained_model,
              ds: Dataset,
              bs=7000):
    plt.figure()
    ax = plt.gca()
    for trained_embs_batch, labels_batch in get_batch_embeddings(trained_model, ds, bs):
        x = trained_embs_batch[:, 0]
        y = trained_embs_batch[:, 1]
        ax.scatter(x.cpu(), y.cpu(), c=labels_batch, s=8, cmap="hsv", alpha=0.6)
    ax.set_title("Before training")
    ax.set_title("After training")
    plt.suptitle("Final embedding space")
    plt.show()


if __name__ == '__main__':

    # Defining configuration
    if config.seed:
        torch.manual_seed(config.seed)
    dev = torch.device(config.dev)
    if args.perplexity is not None:
        config.training_params["perplexity"] = args.perplexity
    if args.epochs is not None:
        config.training_params["n_epochs"] = args.epochs
    if args.batchsize is not None:
        config.training_params["batch_size"] = args.batchsize

    print(f"Training on {dev}", flush=True)

    # Defining dataset
    points, labels = load_mnist_some_classes('data/mnist.pkl.gz', None)
    dim_input = points.size(1)
    points = points.to(dev)
    points_ds = TensorDataset(points, labels)

    train_parametric_tsne_model(points_ds, dim_input, config)

    # Plotting results
    # plot_embs(ffnn, points_ds, config.training_params["batch_size"])
