import torch
from torch.nn.modules import loss
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import numpy as np

import os
import shutil

from offline_scheduler import Scheduler
from models.beta_vae import VAE
from models.resnet import ResNet50, reg_loss_fn, ce_loss_fn
from models.FC import FC
from models.SegmentVGG16 import FeatureExtractor as FE, Classifier as CF
from models.decoder import TCNND, AdversaryModelGen
from utils import copy_source_code
from config_utils import load_config

def run_experiment():
    config = load_config(offline=True)
    gpu_devices = config.get("gpu_devices")
    gpu_id = gpu_devices[0]
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(gpu_id))
    else:
        device = torch.device('cpu')
    seed = config.get("seed")
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
    # lr should eventually move to method once we do some hparam tuning
    lr = config.get("lr")

    # set up paths
    experiment_name = config.get("experiment_name")
    results_path = config.get("results_path_base") + experiment_name
    log_path = results_path + "/logs/"
    model_path = results_path + "/saved_models"

    train_split = config.get("train_split")
    train_batch_size = config.get("train_batch_size_per_gpu") * len(gpu_devices)
    test_batch_size = config.get("test_batch_size")

    method = config.get("method")
    is_cas = config.get("is_cas")

    if method == "ours":
        # Sanitizer params
        dcorr = config.get("dcorr")
        non_priv_pred = config.get("adv")
        priv_dim = config.get("priv_dim")
        alpha1, alpha2, alpha3, alpha4 = config.get("alpha_values")
        components = config.get("sensitive_components")
        # VAE params
        nz = config.get("nz")
        beta = config.get("beta")
        nc = config.get("nc")
        ngf = config.get("ngf")
        ndf = config.get("ndf")
        mechanism = config.get("mechanism")
        hparams_vae = {"nc": nc, "ngf": ngf, "ndf": ndf, "nz": nz}

        vae = VAE(hparams_vae)
        vae.to(device)
        multi_gpu_vae = nn.DataParallel(vae, device_ids=gpu_devices)

        hparams_priv_pred = {"inp_dim": priv_dim, "logits": logits}
        #priv_pred_model = FC(hparams_priv_pred).to(device)
        #priv_pred_model = nn.DataParallel(priv_pred_model, device_ids=gpu_devices)

        if mechanism == "suppression":
            hparams_model = {"inp_dim": nz - priv_dim, "logits": logits}
            model = FC(hparams_model)
        elif mechanism in ["suppression_gen", "sampling", "obfuscation", "dpgmm"]:
            hparams_model = {"logits": logits}
            model = CF(attribute, logits, split_layer=0)
        else:
            print("unknown mechanism {}".format(mechanism))
        model.to(device)
        multi_gpu_model = nn.DataParallel(model, device_ids=gpu_devices)

        model_optim = optim.Adam(multi_gpu_model.parameters(), lr=lr)

        objects = {"vae": multi_gpu_vae, "model_optim": model_optim, "device": device,
                   "model": model, "pred_loss_fn": ce_loss_fn, "alpha1": alpha1, "alpha2": alpha2,
                   "alpha3": alpha3, "alpha4": alpha4, "priv_dim": priv_dim, "beta": beta, "logits": logits,
                   "dcorr": dcorr, "non_priv_pred": non_priv_pred, "mechanism": mechanism, "components": components}
        if mechanism in ["obfuscation", "dpgmm"]:
            eps = config.get("eps")
            objects.update({"eps": eps})
            
    elif method == "vae":
        # VAE params
        nz = config.get("nz")
        beta = config.get("beta")
        nc = config.get("nc")
        ngf = config.get("ngf")
        ndf = config.get("ndf")
        hparams_vae = {"nc": nc, "ngf": ngf, "ndf": ndf, "nz": nz}

        vae = VAE(hparams_vae)
        vae.to(device)
        multi_gpu_vae = nn.DataParallel(vae, device_ids=gpu_devices)

        hparams_model = {"inp_dim": nz, "logits": logits}
        model = FC(hparams_model)
        model.to(device)
        multi_gpu_model = nn.DataParallel(model, device_ids=gpu_devices)

        model_optim = optim.Adam(multi_gpu_model.parameters(), lr=lr)

        objects = {"vae": multi_gpu_vae, "model_optim": model_optim, "device": device,
                   "model": model, "pred_loss_fn": ce_loss_fn, "nz": nz}
        
    elif method in ["tiprdc", "adversarial", "maxentropy"]:
        loss_fn = ce_loss_fn
        encoder = nn.DataParallel(FE().to(device), device_ids=gpu_devices)
        decoder = nn.DataParallel(AdversaryModelGen({"channels": 128, "downsampling": 4, "offset": 1}).to(device), device_ids=gpu_devices)
        if is_cas:
            classifier = nn.DataParallel(CF(logits, split_layer=0).to(device), device_ids=gpu_devices)
        else:
            classifier = nn.DataParallel(CF(attribute, logits).to(device), device_ids=gpu_devices)
        optim_classifier = optim.Adam(classifier.parameters(), lr=lr)

        objects = {"encoder": encoder, "classifier": classifier,
                   "optim_classifier": optim_classifier, "decoder": decoder,
                   "loss_fn": loss_fn, "device": device}

    elif method == "noise":
        loss_fn = ce_loss_fn
        sigma = config.get("sigma")
        classifier = nn.DataParallel(CF(logits, split_layer=0).to(device), device_ids=gpu_devices)
        optim_classifier = optim.Adam(classifier.parameters(), lr=lr)
        objects = {"classifier": classifier, "optim_classifier": optim_classifier,
                   "loss_fn": loss_fn, "device": device, "sigma": sigma}

    elif method == "gap":
        loss_fn = ce_loss_fn
        decoder = nn.DataParallel(TCNND().to(device), device_ids=gpu_devices)
        classifier = nn.DataParallel(CF(attribute, logits, split_layer=0).to(device), device_ids=gpu_devices)
        optim_classifier = optim.Adam(classifier.parameters(), lr=lr)

        objects = {"decoder": decoder, "classifier": classifier,
                   "optim_classifier": optim_classifier,
                   "loss_fn": loss_fn, "device": device}

    elif method == "noprivacy":
        pass

    scheduler_config = {"method": method, "experiment_name": experiment_name, "dataset_path": dataset_path,
              "total_epochs": total_epochs, "train_split": train_split,
              "train_batch_size": train_batch_size, "dataset": dataset,
              "test_batch_size": test_batch_size, "log_path": log_path,
              "model_path": model_path, "im_size": im_size, "attribute": attribute,
              "split": split_data, "is_cas": is_cas} 

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
