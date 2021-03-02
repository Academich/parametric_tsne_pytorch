import datetime
import json
from math import log2
from typing import Optional
import os

import torch

from torch import Tensor, tensor, eye, device, ones, isnan, zeros
from torch import max as torch_max
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from numpy import array
from numpy import save as np_save
from tqdm import tqdm

from ptsne.utils import get_random_string, entropy, distance_functions
from config import config

EPS = tensor([1e-10]).to(device(config.dev))


def calculate_optimized_p_cond(input_points: tensor,
                               target_entropy: float,
                               dist_func: str,
                               tol: float,
                               max_iter: int,
                               min_allowed_sig_sq: float,
                               max_allowed_sig_sq: float,
                               dev: str) -> Optional[tensor]:
    """
    Calculates conditional probability matrix optimized by binary search
    :param input_points: A matrix of input data where every row is a data point
    :param target_entropy: The entropy that every distribution (row) in conditional
    probability matrix will be optimized to match
    :param dist_func: A name for desirable distance function (e.g. "euc", "jaccard" etc)
    :param tol: A small number - tolerance threshold for binary search
    :param max_iter: Maximum number of binary search iterations
    :param min_allowed_sig_sq: Minimum allowed value for the spread of any distribution
    in conditional probability matrix
    :param max_allowed_sig_sq: Maximum allowed value for the spread of any distribution
    in conditional probability matrix
    :param dev: device for tensors (e.g. "cpu" or "cuda")
    :return:
    """
    n_points = input_points.size(0)

    # Calculating distance matrix with the given distance function
    dist_f = distance_functions[dist_func]
    distances = dist_f(input_points)
    diag_mask = (1 - eye(n_points)).to(device(dev))

    # Initializing sigmas
    min_sigma_sq = (min_allowed_sig_sq + 1e-20) * ones(n_points).to(device(dev))
    max_sigma_sq = max_allowed_sig_sq * ones(n_points).to(device(dev))
    sq_sigmas = (min_sigma_sq + max_sigma_sq) / 2

    # Computing conditional probability matrix from distance matrix
    p_cond = get_p_cond(distances, sq_sigmas, diag_mask)

    # Making a vector of differences between target entropy and entropies for all rows in p_cond
    ent_diff = entropy(p_cond) - target_entropy

    # Binary search ends when all entropies match the target entropy
    finished = ent_diff.abs() < tol

    curr_iter = 0
    while not finished.all().item():
        if curr_iter >= max_iter:
            print("Warning! Exceeded max iter.", flush=True)
            # print("Discarding batch")
            return p_cond
        pos_diff = (ent_diff > 0).float()
        neg_diff = (ent_diff <= 0).float()

        max_sigma_sq = pos_diff * sq_sigmas + neg_diff * max_sigma_sq
        min_sigma_sq = pos_diff * min_sigma_sq + neg_diff * sq_sigmas

        sq_sigmas = finished.logical_not() * (min_sigma_sq + max_sigma_sq) / 2 + finished * sq_sigmas
        p_cond = get_p_cond(distances, sq_sigmas, diag_mask)
        ent_diff = entropy(p_cond) - target_entropy
        finished = ent_diff.abs() < tol
        curr_iter += 1
    if isnan(ent_diff.max()):
        print("Warning! Entropy is nan. Discarding batch", flush=True)
        return
    return p_cond


def get_p_cond(distances: tensor, sigmas_sq: tensor, mask: tensor) -> tensor:
    """
    Calculates conditional probability distribution given distances and squared sigmas
    :param distances: Matrix of squared distances ||x_i - x_j||^2
    :param sigmas_sq: Row vector of squared sigma for each row in distances
    :param mask: A mask tensor to set diagonal elements to zero
    :return: Conditional probability matrix
    """
    logits = -distances / (2 * torch_max(sigmas_sq, EPS).view(-1, 1))
    logits.exp_()
    masked_exp_logits = logits * mask
    normalization = torch_max(masked_exp_logits.sum(1), EPS).unsqueeze(1)
    return masked_exp_logits / normalization + 1e-10


def get_q_joint(emb_points: tensor, dist_func: str, alpha: int, ) -> tensor:
    """
    Calculates the joint probability matrix in embedding space.
    :param emb_points: Points in embeddings space
    :param alpha: Number of degrees of freedom in t-distribution
    :param dist_func: A kay name for a distance function
    :return: Joint distribution matrix in emb. space
    """
    n_points = emb_points.size(0)
    mask = (-eye(n_points) + 1).to(emb_points.device)
    dist_f = distance_functions[dist_func]
    distances = dist_f(emb_points) / alpha
    q_joint = (1 + distances).pow(-(1 + alpha) / 2) * mask
    q_joint /= q_joint.sum()
    return torch_max(q_joint, EPS)


def make_joint(distr_cond: tensor) -> tensor:
    """
    Makes a joint probability distribution out of conditional distribution
    :param distr_cond: Conditional distribution matrix
    :return: Joint distribution matrix. All values in it sum up to 1.
    Too small values are set to fixed epsilon
    """
    n_points = distr_cond.size(0)
    distr_joint = (distr_cond + distr_cond.t()) / (2 * n_points)
    return torch_max(distr_joint, EPS)


def loss_function(p_joint: Tensor, q_joint: Tensor) -> Tensor:
    """
    Calculates KLDiv between joint distributions in original and embedding space
    :param p_joint:
    :param q_joint:
    :return: KLDiv value
    """
    # TODO Add here alpha gradient calculation too
    # TODO Add L2-penalty for early compression?
    return (p_joint * torch.log((p_joint + EPS) / (q_joint + EPS))).sum()


def fit_ptsne_model(model: torch.nn.Module,
                    input_points: Dataset,
                    opt: Optimizer,
                    perplexity: Optional[int],
                    n_epochs: int,
                    dev: str,
                    save_dir_path: str,
                    epochs_to_save_after: Optional[int],
                    early_exaggeration: int,
                    early_exaggeration_constant: int,
                    batch_size: int,
                    dist_func_name: str,
                    bin_search_tol: float,
                    bin_search_max_iter: int,
                    min_allowed_sig_sq: float,
                    max_allowed_sig_sq: float,
                    configuration_report: str
                    ) -> None:
    """
    Fits a parametric t-SNE model and optionally saves it to the desired directory.
    Fits either regular or multi-scale t-SNE
    :param model: nn.Module instance
    :param input_points: tensor of original points
    :param opt: optimizer instance
    :param perplexity: perplexity of a model. If passed None, multi-scale parametric t-SNE
    model will be trained
    :param n_epochs: Number of epochs for training
    :param dev: device for tensors (e.g. "cpu" or "cuda")
    :param save_dir_path: path to directory to save a trained model to
    :param epochs_to_save_after: number of epochs to save a model after. If passed None,
    model won't be saved at all
    :param early_exaggeration: Number of first training cycles in which
    exaggeration will be applied
    :param early_exaggeration_constant: Constant by which p_joint is multiplied in early exaggeration
    :param batch_size: Batch size for training
    :param dist_func_name: Name of distance function for distance matrix.
    Possible names: "euc", "jaccard", "cosine"
    :param bin_search_tol: A small number - tolerance threshold for binary search
    :param bin_search_max_iter: Number of max iterations for binary search
    :param min_allowed_sig_sq: Minimum allowed value for the spread of any distribution
    in conditional probability matrix
    :param max_allowed_sig_sq: Maximum allowed value for the spread of any distribution
    in conditional probability matrix
    :param configuration_report: Config of the model in string form for report purposes
    :return:
    """
    model.train()
    batches_passed = 0
    model_name = get_random_string(6)
    epoch_losses = []

    # Function operates with DataLoader
    train_dl = DataLoader(input_points, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        train_loss = 0
        epoch_start_time = datetime.datetime.now()

        # For every batch
        for list_with_batch in tqdm(train_dl):
            orig_points_batch, _ = list_with_batch

            # Calculate conditional probability matrix in higher-dimensional space for the batch

            # Regular parametric t-SNE
            if perplexity is not None:
                target_entropy = log2(perplexity)
                p_cond_in_batch = calculate_optimized_p_cond(orig_points_batch,
                                                             target_entropy,
                                                             dist_func_name,
                                                             bin_search_tol,
                                                             bin_search_max_iter,
                                                             min_allowed_sig_sq,
                                                             max_allowed_sig_sq,
                                                             dev)
                if p_cond_in_batch is None:
                    continue
                p_joint_in_batch = make_joint(p_cond_in_batch)

            # Multiscale parametric t-SNE
            else:
                max_entropy = round(log2(batch_size / 2))
                n_different_entropies = 0
                mscl_p_joint_in_batch = zeros(batch_size, batch_size).to(device(dev))
                for h in range(1, max_entropy):
                    p_cond_for_h = calculate_optimized_p_cond(orig_points_batch,
                                                              h,
                                                              dist_func_name,
                                                              bin_search_tol,
                                                              bin_search_max_iter,
                                                              min_allowed_sig_sq,
                                                              max_allowed_sig_sq,
                                                              dev)
                    if p_cond_for_h is None:
                        continue
                    n_different_entropies += 1

                    p_joint_for_h = make_joint(p_cond_for_h)

                    # TODO This fails if the last batch doesn't match the shape of mscl_p_joint_in_batch
                    mscl_p_joint_in_batch += p_joint_for_h

                p_joint_in_batch = mscl_p_joint_in_batch / n_different_entropies

            # Apply early exaggeration to the conditional probability matrix
            if early_exaggeration:
                p_joint_in_batch *= early_exaggeration_constant
                early_exaggeration -= 1

            batches_passed += 1
            opt.zero_grad()

            # Calculate joint probability matrix in lower-dimensional space for the batch
            embeddings = model(orig_points_batch)
            q_joint_in_batch = get_q_joint(embeddings, "euc", alpha=1)

            # Calculate loss
            loss = loss_function(p_joint_in_batch, q_joint_in_batch)
            train_loss += loss.item()

            # Make an optimization step
            loss.backward()
            opt.step()

        epoch_end_time = datetime.datetime.now()
        time_elapsed = epoch_end_time - epoch_start_time

        # Report loss for epoch
        average_loss = train_loss / batches_passed
        epoch_losses.append(average_loss)
        print(f'====> Epoch: {epoch + 1}. Time {time_elapsed}. Average loss: {average_loss:.4f}', flush=True)

        # Save model and loss history if needed
        save_path = os.path.join(save_dir_path, f"{model_name}_epoch_{epoch + 1}")
        if epochs_to_save_after is not None and (epoch + 1) % epochs_to_save_after == 0:
            torch.save(model, save_path + ".pt")
            with open(save_path + ".json", "w") as here:
                json.dump(json.loads(configuration_report), here)
            print('Model saved as %s' % save_path, flush=True)

        if epochs_to_save_after is not None and epoch == n_epochs - 1:
            epoch_losses = array(epoch_losses)
            loss_save_path = save_path + "_loss.npy"
            np_save(loss_save_path, epoch_losses)
            print("Loss history saved in", loss_save_path, flush=True)
