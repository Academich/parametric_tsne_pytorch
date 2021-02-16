import json
import datetime

import torch

from ptsne.ptsne_model import NeuralMapper
from ptsne.ptsne_training import fit_ptsne_model


def train_parametric_tsne_model(points_ds, input_dimens, config):
    net = NeuralMapper
    ffnn = net(dim_input=input_dimens).to(torch.device(config.dev))
    opt = torch.optim.Adam(ffnn.parameters(), **config.optimization_conf)

    report_config = json.dumps(
        {"device": config.dev,
         "seed": config.seed,
         "optimization": config.optimization_conf,
         "training": config.training_params})

    start = datetime.datetime.now()

    fit_ptsne_model(ffnn,
                    points_ds,
                    opt,
                    **config.training_params,
                    epochs_to_save_after=config.epochs_to_save_after,
                    dev=config.dev,
                    save_dir_path=config.save_dir_path,
                    configuration_report=report_config)

    fin = datetime.datetime.now()
    print("Training time:", fin - start, flush=True)
