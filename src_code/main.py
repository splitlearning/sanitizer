import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import numpy as np

import os
import shutil

from scheduler import Scheduler
from models.beta_vae import VAE, loss_function as vae_loss_fn
from models.resnet import ResNet50, reg_loss_fn, ce_loss_fn, DistortionLoss, DistCorrelation
from models.simple_encoder import SimpleEncoder, EntropyLoss as entropy_loss
from models.FC import FC
from models.SegmentVGG16 import FeatureExtractor as FE, Classifier as CF, reconstruction_loss
from models.MutualInformation import MutlInfo, info_loss
from models.decoder import AdversaryModelGen, TCNND
from utils import copy_source_code
from config_utils import load_config
import random


def run_experiment():
    config = load_config(offline=False)
    seed = config.get("seed")
    gpu_devices = config.get("gpu_devices")
    gpu_id = gpu_devices[0]
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(gpu_id))
    else:
        device = torch.device('cpu')
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    total_epochs = config.get("total_epochs")
    dataset = config.get("dataset")
    dataset_path = config.get("dataset_path")
    im_size = config.get("im_size")
    attribute = config.get("attribute")
    if type(attribute) == list and len(attribute) == 1:
        attribute = attribute[0]
    split_data = config.get("split_data")
    split = config.get("split")
    logits = config.get("logits")

    # set up paths
    experiment_name = config.get("experiment_name")
    results_path = config.get("results_path_base") + experiment_name
    log_path = results_path + "/logs/"
    model_path = results_path + "/saved_models"
    # lr should eventually move to method once we do some hparam tuning
    lr = config.get("lr")

    train_split = config.get("train_split")
    train_batch_size = config.get("train_batch_size_per_gpu") * len(gpu_devices)
    test_batch_size = config.get("test_batch_size")

    method = config.get("method")

    if method == "ours":
        # DIVR params
        dcorr = config.get("dcorr")
        adv = config.get("adv")
        priv_dim = config.get("priv_dim")
        alpha1, alpha2, alpha3, alpha4 = config.get("alpha_values")
        loss_fn = ce_loss_fn

        # VAE params
        nz = config.get("nz")
        beta = config.get("beta")
        nc = config.get("nc")
        ngf = config.get("ngf")
        ndf = config.get("ndf")
        hparams_vae = {"nc": nc, "ngf": ngf, "ndf": ndf, "nz": nz}

	    # initialize models
        vae = VAE(hparams_vae).to(device)
        multi_gpu_vae = nn.DataParallel(vae, device_ids=gpu_devices)

        hparams_aligner = {"inp_dim": priv_dim, "logits": logits}
        aligner_model = FC(hparams_aligner).to(device)
        aligner_model = nn.DataParallel(aligner_model, device_ids=gpu_devices)

        hparams_adv = {"inp_dim": nz - priv_dim, "logits": logits}
        adv_model = FC(hparams_adv).to(device)
        adv_model = nn.DataParallel(adv_model, device_ids=gpu_devices)

	    # optimizers
        joint_params = list(aligner_model.parameters()) +\
                       list(multi_gpu_vae.parameters())
        joint_optim = optim.Adam(joint_params, lr=lr)
        adv_optim = optim.Adam(adv_model.parameters(), lr=lr)

        objects = {"vae": multi_gpu_vae, "vae_loss_fn": vae_loss_fn, "device": device,
                   "pred_loss_fn": ce_loss_fn, "alpha1": alpha1, "alpha2": alpha2,
                   "alpha3": alpha3, "alpha4": alpha4, "priv_dim": priv_dim, "beta": beta,
                   "dcorr": dcorr, "adv_model": adv_model, "aligner": aligner_model,
                   "adv_optim": adv_optim, "joint_optim": joint_optim, "dcorr_fn": DistCorrelation(),
                   "adv": adv}

    elif method == "vae":
        # VAE params
        nz = config.get("nz")
        beta = config.get("beta")
        nc = config.get("nc")
        ngf = config.get("ngf")
        ndf = config.get("ndf")
        hparams_vae = {"nc": nc, "ngf": ngf, "ndf": ndf, "nz": nz}

	    # initialize models
        vae = VAE(hparams_vae).to(device)
        multi_gpu_vae = nn.DataParallel(vae, device_ids=gpu_devices)

	    # optimizers
        optim_vae = optim.Adam(multi_gpu_vae.parameters(), lr=lr)

        objects = {"vae": multi_gpu_vae, "vae_loss_fn": vae_loss_fn, "device": device,
                   "beta": beta, "optim": optim_vae}

    elif method == "tiprdc":
        _lambda = config.get("lambda")
        mi_estimator = nn.DataParallel(MutlInfo(logits).to(device), device_ids=gpu_devices)
        mi_loss = info_loss
        loss_fn = ce_loss_fn
        encoder = nn.DataParallel(FE().to(device), device_ids=gpu_devices)
        classifier = nn.DataParallel(CF(logits).to(device),device_ids=gpu_devices)
        optim_estimator = optim.Adam(mi_estimator.parameters(), lr=lr)
        optim_encoder = optim.Adam(encoder.parameters(), lr=lr)
        optim_classifier = optim.Adam(classifier.parameters(), lr=lr)

        objects = {"mi_estimator": mi_estimator, "mi_loss": mi_loss, "encoder": encoder, "classifier": classifier,
                   "optim_estimator": optim_estimator, "optim_encoder": optim_encoder, "optim_classifier": optim_classifier,
                   "loss_fn": loss_fn, "device": device, "lambda": _lambda}

    elif method == "adversarial":
        _lambda = config.get("lambda")
        loss_fn = ce_loss_fn
        decoder = nn.DataParallel(AdversaryModelGen({"channels": 128}).to(device), device_ids=gpu_devices)
        encoder = nn.DataParallel(FE().to(device), device_ids=gpu_devices)
        classifiers = CF(attribute, logits)
        # new_classifiers = dict()
        for attr in classifiers.models.keys():
            classifiers.models[attr] = nn.DataParallel(classifiers.models[attr].to(device),device_ids=gpu_devices)
            # new_classifiers[attr] = classifier
        optim_ae = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
        optim_classifiers = dict()
        for attr in classifiers.models.keys():
            optim_classifier = optim.Adam(classifiers.models[attr].parameters(), lr=lr)
            optim_classifiers[attr] = optim_classifier

        objects = {"decoder": decoder, "reconstruction_loss": reconstruction_loss, "encoder": encoder, "classifier": classifiers,
                   "optim_ae": optim_ae, "optim_classifier": optim_classifiers,
                   "loss_fn": loss_fn, "device": device, "lambda": _lambda}
    elif method == "maxentropy":
        _lambda = config.get("lambda")
        encoder = nn.DataParallel(FE().to(device), device_ids=gpu_devices)
        loss_fn = ce_loss_fn
        decoder = nn.DataParallel(AdversaryModelGen({"channels": 128}).to(device), device_ids=gpu_devices)
        classifier = nn.DataParallel(CF(logits).to(device),device_ids=gpu_devices)
        optim_encoder = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=3e-4)
        optim_classifier = optim.Adam(classifier.parameters(), lr=3e-4)

        objects = {"decoder": decoder, "entropy_loss": entropy_loss(), "encoder": encoder, "classifier": classifier,
                   "optim_encoder": optim_encoder, "optim_classifier": optim_classifier,
                   "loss_fn": loss_fn, "device": device, "lambda": _lambda, "reconstruction_loss": reconstruction_loss}

    elif method == "noise":
        sigma = config.get("sigma")
        loss_fn = ce_loss_fn
        classifier = nn.DataParallel(CF(attribute, logits, 0).to(device), device_ids=gpu_devices)
        optim_classifier = optim.Adam(classifier.parameters(), lr=lr)
        objects = {"classifier": classifier, "optim_classifier": optim_classifier,
                   "loss_fn": loss_fn, "device": device, "sigma": sigma}

    elif method == "gap":
        _lambda = config.get("lambda")
        D = config.get("D")
        loss_fn = ce_loss_fn
        distortion_loss_fn = DistortionLoss()
        decoder = nn.DataParallel(TCNND().to(device), device_ids=gpu_devices)
        classifier = nn.DataParallel(CF(logits, split_layer=0).to(device),device_ids=gpu_devices)
        optim_decoder = optim.Adam(decoder.parameters(), lr=3e-4)
        optim_classifier = optim.Adam(classifier.parameters(), lr=3e-4)

        objects = {"decoder": decoder, "distortion_loss_fn": distortion_loss_fn, "classifier": classifier,
                   "optim_decoder": optim_decoder, "optim_classifier": optim_classifier,
                   "loss_fn": loss_fn, "device": device, "lambda": _lambda, "D": D}
    elif method == "noprivacy":
        loss_fn = ce_loss_fn
        classifier = nn.DataParallel(CF(logits, 0), device_ids=gpu_devices)
        optim_classifier = optim.Adam(classifier.parameters(), lr=lr)
        objects = {"classifier": classifier, "optim_classifier": optim_classifier,
                    "loss_fn": loss_fn, "device": device}
    else:
        print("unknown method; exiting")
        exit()

    scheduler_config = {"method": method, "experiment_name": experiment_name, "dataset_path": dataset_path,
              "total_epochs": total_epochs, "train_split": train_split, "logits": logits,
              "train_batch_size": train_batch_size, "dataset": dataset,
              "test_batch_size": test_batch_size, "log_path": log_path,
              "model_path": model_path, "im_size": im_size, "attribute": attribute,
              "split": split_data} 

    if os.path.isdir(results_path):
        print("Experiment {} already present".format(experiment_name))
        inp = input("Press e to exit, r to replace it: ")
        if inp == "e":
            exit()
        elif inp == "r":
            shutil.rmtree(results_path)
        else:
            print("Input not understood")
            exit()
    copy_source_code(results_path)
    os.mkdir(model_path)
    os.mkdir(log_path)

    scheduler = Scheduler(scheduler_config, objects)

    print("starting {}".format(experiment_name))
    for epoch in range(scheduler_config.get("total_epochs")):
        scheduler.train()
        scheduler.test()

if __name__ == '__main__':
    run_experiment()
