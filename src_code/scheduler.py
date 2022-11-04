import torch
from torchvision import datasets, transforms
import dataset_utils
import numpy as np
from torch.autograd import Variable
import os
from utils import LoggerUtils

class Scheduler():
    """docstring for Scheduler"""

    def __init__(self, config, objects):
        super(Scheduler, self).__init__()
        self.config = config
        self.objects = objects
        self.epoch = 0
        self.initialize()

    def initialize(self):
        self.setup_data_pipeline()
        self.setup_training_params()
        log_config = {"log_path": self.config["log_path"]}
        self.logger = LoggerUtils(log_config)

    def get_split(self, dataset):

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(self.config["train_split"] * dataset_size))
        np.random.shuffle(indices)

        train_indices, test_indices = indices[:split], indices[split:]
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        return train_dataset, test_dataset

    def setup_data_pipeline(self):
        self.IM_SIZE = self.config["im_size"]
        trainTransform = transforms.Compose([
            transforms.Resize((self.IM_SIZE, self.IM_SIZE)),
            transforms.ToTensor()])

        train_config = {"transforms": trainTransform,
                        "path": self.config["dataset_path"],
                        "attribute": self.config["attribute"]}

        dset_name = self.config["dataset"].lower()
        if dset_name == "celeba":
            train_config["train"] = True
            train_dataset = dataset_utils.CelebA(train_config)
            train_config["train"] = False
            test_dataset = dataset_utils.CelebA(train_config)
        elif dset_name == "utkface":
            train_config["format"] = "jpg"
            dataset = dataset_utils.UTKFace(train_config)
        elif dset_name == "fairface":
            train_config["format"] = "jpg"
            train_config["train"] = True
            train_dataset = dataset_utils.FairFace(train_config)
            train_config["train"] = False
            test_dataset = dataset_utils.FairFace(train_config)

        if self.config["split"] is True:
            train_dataset, test_dataset = self.get_split(dataset)

        self.trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config["train_batch_size"],
            shuffle=False, num_workers=5)
        self.dataset_size = len(train_dataset)

        self.testloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.config["test_batch_size"],
            shuffle=False, num_workers=5)

    def setup_training_params(self):
        if self.config["method"] == "ours":
            self.ours_training_params()
        elif self.config["method"] == "tiprdc":
            self.tiprdc_training_params()
        elif self.config["method"] == "adversarial":
            self.adversarial_training_params()
        elif self.config["method"] == "maxentropy":
            self.maxentropy_training_params()
        elif self.config["method"] == "gap":
            self.gap_training_params()
        elif self.config["method"] == "noprivacy":
            self.noprivacy_training_params()
        elif self.config["method"] == "noise":
            self.noise_training_params()
        elif self.config["method"] == "vae":
            self.vae_training_params()
        else:
            exit()

    def vae_training_params(self):
        self.epoch = 0
        self.beta = self.objects["beta"]
        self.vae = self.objects["vae"]
        self.vae_loss_fn = self.objects["vae_loss_fn"]
        self.optim = self.objects["optim"]
        self.device = self.objects["device"]
        self.vae_model_path = self.config["model_path"] + "/model_vae.pt"

    def noise_training_params(self):
        self.epoch = 0
        self.sigma = self.objects["sigma"]
        self.classifier = self.objects["classifier"]
        self.optim_classifier = self.objects["optim_classifier"]
        self.device = self.objects["device"]
        self.loss_fn = self.objects["loss_fn"]
        self.classifier_path = self.config["model_path"] + "/classifier.pt"

    def noprivacy_training_params(self):
        self.epoch = 0
        self.classifier = self.objects["classifier"]
        self.optim_classifier = self.objects["optim_classifier"]
        self.device = self.objects["device"]
        self.loss_fn = self.objects["loss_fn"]
        self.classifier_path = self.config["model_path"] + "/classifier.pt"

    def gap_training_params(self):
        self.epoch = 0
        self.decoder = self.objects["decoder"]
        self.classifier = self.objects["classifier"]

        self.optim_decoder = self.objects["optim_decoder"]
        self.optim_classifier = self.objects["optim_classifier"]

        self.device = self.objects["device"]
        self.loss_fn = self.objects["loss_fn"]
        self.distortion_loss_fn = self.objects["distortion_loss_fn"]
        self.lambda_ = self.objects["lambda"]
        self.D = self.objects["D"]

        self.decoder_path = self.config["model_path"] + "/decoder.pt"
        self.classifier_path = self.config["model_path"] + "/classifier.pt"

    def maxentropy_training_params(self):
        self.epoch = 0
        self.decoder = self.objects["decoder"]
        self.encoder = self.objects["encoder"]
        self.classifier = self.objects["classifier"]

        self.optim_encoder = self.objects["optim_encoder"]
        self.optim_classifier = self.objects["optim_classifier"]

        self.device = self.objects["device"]
        self.entropy_loss = self.objects["entropy_loss"]
        self.loss_fn = self.objects["loss_fn"]
        self.reconstruction_loss = self.objects["reconstruction_loss"]
        self.lambda_ = self.objects["lambda"]

        self.decoder_path = self.config["model_path"] + "/decoder.pt"
        self.encoder_path = self.config["model_path"] + "/encoder.pt"
        self.classifier_path = self.config["model_path"] + "/classifier.pt"

    def adversarial_training_params(self):
        self.epoch = 0
        self.decoder = self.objects["decoder"]
        self.encoder = self.objects["encoder"]
        self.classifier = self.objects["classifier"]

        self.optim_ae = self.objects["optim_ae"]
        self.optim_classifier = self.objects["optim_classifier"]

        self.device = self.objects["device"]
        self.reconstruction_loss = self.objects["reconstruction_loss"]
        self.loss_fn = self.objects["loss_fn"]
        self.lambda_ = self.objects["lambda"]

        self.decoder_path = self.config["model_path"] + "/decoder.pt"
        self.encoder_path = self.config["model_path"] + "/encoder.pt"
        self.classifier_path = self.config["model_path"] + "/classifier.pt"

    def tiprdc_training_params(self):
        self.epoch = 0
        self.mi_estimator = self.objects["mi_estimator"]
        self.encoder = self.objects["encoder"]
        self.classifier = self.objects["classifier"]

        self.optim_estimator = self.objects["optim_estimator"]
        self.optim_encoder = self.objects["optim_encoder"]
        self.optim_classifier = self.objects["optim_classifier"]

        self.device = self.objects["device"]
        self.mi_loss = self.objects["mi_loss"]
        self.loss_fn = self.objects["loss_fn"]
        self.lambda_ = self.objects["lambda"]

        self.estimator_path = self.config["model_path"] + "/estimator.pt"
        self.encoder_path = self.config["model_path"] + "/encoder.pt"
        self.classifier_path = self.config["model_path"] + "/classifier.pt"

    def ours_training_params(self):
        self.epoch = 0
        self.beta = self.objects["beta"]
        # alpha 1->VAE, 2->Aligner, 3->Adv, 4->Dcorr
        self.alpha1, self.alpha2, self.alpha3, self.alpha4 = self.objects["alpha1"], self.objects["alpha2"], self.objects["alpha3"], self.objects["alpha4"]
        self.dcorr = self.objects["dcorr"]
        self.adv = self.objects["adv"]
        if self.dcorr:
            self.dcorr_fn = self.objects["dcorr_fn"]
        self.adv_model = self.objects["adv_model"]
        self.priv_dim = self.objects["priv_dim"]
        self.vae = self.objects["vae"]
        self.aligner = self.objects["aligner"]

        self.vae_loss_fn = self.objects["vae_loss_fn"]
        self.aligner_loss_fn = self.objects["pred_loss_fn"]
        self.adv_loss_fn = self.objects["pred_loss_fn"]
        # Performing joint optimization right now
        self.optim = self.objects["joint_optim"]
        self.adv_optim = self.objects["adv_optim"]
        self.device = self.objects["device"]

        self.vae_model_path = self.config["model_path"] + "/model_vae.pt"
        self.aligner_model_path = self.config["model_path"] + "/model_aligner.pt"
        self.adv_model_path = self.config["model_path"] + "/model_adv.pt"

    def test(self):
        if self.config["method"] == "ours":
            self.ours_test()
        elif self.config["method"] == "tiprdc":
            self.tiprdc_test()
        elif self.config["method"] == "adversarial":
            self.adversarial_test()
        elif self.config["method"] == "maxentropy":
            self.maxentropy_test()
        elif self.config["method"] == "gap":
            self.gap_test()
        elif self.config["method"] == "noprivacy":
            self.noprivacy_test()
        elif self.config["method"] == "noise":
            self.noise_test()
        elif self.config["method"] == "vae":
            self.vae_test()

    def vae_test(self):
        self.vae.eval()
        save_image_interval = 200
        test_loss = 0
        total = 0
        os.mkdir("{}/{}".format(self.config["log_path"], self.epoch))

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(self.testloader):
                data = Variable(data)
                data = data.to(self.device)
                labels = Variable(labels).to(self.device)
                recon_batch, mu, logvar, z = self.vae(data)

                vae_loss, rec_loss, kld_loss = self.vae_loss_fn(recon_batch, data, mu, logvar, self.beta)
                test_loss += vae_loss.item()
                total += int(data.shape[0])

                if batch_idx % save_image_interval == 0:
                    filepath = "{}/inp_{}.jpg".format(self.epoch, batch_idx)
                    self.logger.save_image_batch(data.data, filepath)
                    filepath = "{}/out_{}.jpg".format(self.epoch, batch_idx)
                    self.logger.save_image_batch(recon_batch.data, filepath)
        test_loss /= total

        self.logger.log_scalar("test/loss", test_loss, self.epoch)
        self.logger.log_console("epoch {}, average test loss {:.4f}".format(self.epoch,
                                                              test_loss))

    def noise_test(self):
        total = 0
        classifier_loss = 0
        self.classifier.eval()
        total_loss, correct = 0, 0 
        os.mkdir("{}/{}".format(self.config["log_path"], self.epoch))
        for batch_idx, (data, labels) in enumerate(self.testloader):
            data = Variable(data).to(self.device)
            labels = Variable(labels).to(self.device)

            data = data + torch.randn(data.size()).to(self.device) * self.sigma
            pred = self.classifier(data)

            correct += (pred.argmax(dim=1) == labels).sum().item()
            loss = self.loss_fn(pred, labels)

            total += data.shape[0]
            total_loss += loss.item()

        self.logger.log_scalar("test/loss", total_loss / total, self.epoch)
        self.logger.log_scalar("test/non_priv_accuracy", correct / total,
                               self.epoch)
        self.logger.log_console("test epoch {}, loss {:.4f}, accuracy {:.3f}"
                                .format(self.epoch,
                                        total_loss / total,
                                        correct / total))

    def noprivacy_test(self):
        total = 0
        classifier_loss = 0
        self.classifier.eval()
        total_loss, correct = 0, 0 
        os.mkdir("{}/{}".format(self.config["log_path"], self.epoch))
        for batch_idx, (data, labels) in enumerate(self.testloader):
            data = Variable(data).to(self.device)
            labels = Variable(labels).to(self.device)

            pred = self.classifier(data)

            correct += (pred.argmax(dim=1) == labels).sum().item()
            loss = self.loss_fn(pred, labels)

            total += data.shape[0]
            total_loss += loss.item()

        self.logger.log_scalar("test/loss", total_loss / total, self.epoch)
        self.logger.log_scalar("test/non_priv_accuracy", correct / total,
                               self.epoch)
        self.logger.log_console("test epoch {}, loss {:.4f}, accuracy {:.3f}"
                                .format(self.epoch,
                                        total_loss / total,
                                        correct / total))

    def gap_test(self):
        total = 0
        classifier_loss = 0
        save_image_interval = 200
        self.classifier.eval()
        self.decoder.eval()
        total_loss, correct, distortion_loss, ce_loss = 0, 0, 0, 0
        os.mkdir("{}/{}".format(self.config["log_path"], self.epoch))
        for batch_idx, (data, labels) in enumerate(self.testloader):
            data = Variable(data).to(self.device)
            labels = Variable(labels).to(self.device)

            z = torch.randn(data.size(0), 100).to(data.device)
            noise = self.decoder(z)
            x_prime = data + noise
            pred = self.classifier(x_prime)

            correct += (pred.argmax(dim=1) == labels).sum().item()
            loss_ce = self.loss_fn(pred, labels)
            loss_distortion = torch.max(torch.tensor(0.).to(self.device), self.distortion_loss_fn(x_prime, data) - self.D)
            loss =  - self.lambda_ * loss_ce + (1. - self.lambda_) * loss_distortion

            total += data.shape[0]
            total_loss += loss.item()
            ce_loss += loss_ce.item()
            distortion_loss += loss_distortion.item()

            if batch_idx % save_image_interval == 0:
                filepath = "{}/inp_{}.jpg".format(self.epoch, batch_idx)
                self.logger.save_image_batch(data.data, filepath)
                filepath = "{}/out_{}.jpg".format(self.epoch, batch_idx)
                self.logger.save_image_batch(x_prime.data, filepath)

        self.logger.log_scalar("test/loss", total_loss / total, self.epoch)
        self.logger.log_scalar("test/ce_loss", ce_loss / total, self.epoch)
        self.logger.log_scalar("test/distortion_loss", distortion_loss / total,
                               self.epoch)
        self.logger.log_scalar("test/non_priv_accuracy", correct / total,
                               self.epoch)
        self.logger.log_console("test epoch {}, loss {:.4f}, distortion_loss {:.4f}, priv_accuracy {:.3f}, ce_loss {:.3f}"
                                .format(self.epoch,
                                        total_loss / total,
                                        distortion_loss / total,
                                        correct / total,
                                        ce_loss / total))

    def maxentropy_test(self):
        total = 0
        classifier_loss = 0
        save_image_interval = 200
        self.encoder.eval()
        self.classifier.eval()
        self.decoder.eval()
        total_loss, correct, ae_loss, entropy_loss = 0, 0, 0, 0
        os.mkdir("{}/{}".format(self.config["log_path"], self.epoch))
        for batch_idx, (data, labels) in enumerate(self.testloader):
            data = Variable(data).to(self.device)
            labels = Variable(labels).to(self.device)

            z = self.encoder(data)
            pred = self.classifier(z)
            x_prime = self.decoder(z)

            correct += (pred.argmax(dim=1) == labels).sum().item()
            loss_ae = self.reconstruction_loss(data, x_prime)
            loss_entropy = self.entropy_loss(pred)
            loss = -self.lambda_ * loss_entropy + (1. - self.lambda_) * loss_ae

            total += data.shape[0]
            total_loss += loss.item()
            ae_loss += loss_ae.item()
            entropy_loss += loss_entropy.item()

            if batch_idx % save_image_interval == 0:
                filepath = "{}/inp_{}.jpg".format(self.epoch, batch_idx)
                self.logger.save_image_batch(data.data, filepath)
                filepath = "{}/out_{}.jpg".format(self.epoch, batch_idx)
                self.logger.save_image_batch(x_prime.data, filepath)

        self.logger.log_scalar("test/loss", total_loss / total, self.epoch)
        self.logger.log_scalar("test/entropy_loss", entropy_loss / total, self.epoch)
        self.logger.log_scalar("test/ae_loss", ae_loss / total,
                               self.epoch)
        self.logger.log_scalar("test/non_priv_accuracy", correct / total,
                               self.epoch)
        self.logger.log_console("test epoch {}, loss {:.4f}, priv_pred_loss {:.4f}, priv_accuracy {:.3f}, entropy_loss {:.3f}"
                                .format(self.epoch,
                                        total_loss / total,
                                        ae_loss / total,
                                        correct / total,
                                        entropy_loss / total))

    def adversarial_test(self):
        num_logits = self.config["logits"]
        ae_loss = 0
        classifier_loss = 0
        self.encoder.eval()
        self.classifier.eval()
        self.decoder.eval()
        total, total_loss, correct, cf_loss, ae_loss = 0, 0, 0, 0, 0
        ####
        total_loss = dict()
        correct = dict()
        cf_loss = dict()
        classifier_loss = dict()
        for attr in self.classifier.models.keys():
            self.classifier.models[attr].train()
            total_loss[attr] = 0
            correct[attr] = 0
            cf_loss[attr] = 0
            classifier_loss[attr] = 0

        save_image_interval = 200
        os.mkdir("{}/{}".format(self.config["log_path"], self.epoch))
        for batch_idx, (data, labels) in enumerate(self.testloader):
            data = Variable(data).to(self.device)
            labels = Variable(labels).to(self.device)

            z = self.encoder(data)
            pred = self.classifier(z)
            x_prime = self.decoder(z)

            loss_cf = dict()
            for attr in pred.keys():
                label_ind = self.config["attribute"].index(attr)
                label = labels[:, label_ind]
                attr_correct = (pred[attr].argmax(dim=1) == label).sum().item()
                correct[attr] += attr_correct
                attr_loss_cf = self.loss_fn(pred[attr], label.type(torch.LongTensor))
                loss_cf[attr] = attr_loss_cf
            
            loss_ae = self.reconstruction_loss(data, x_prime)
            
            loss = dict()
            attr_total_loss_per_batch = None

            for attr in loss_cf.keys():
                attr_loss = -self.lambda_ * loss_cf[attr] + (1. - self.lambda_) * loss_ae
                loss[attr] = attr_loss
                if attr_total_loss_per_batch is None:
                    attr_total_loss_per_batch = attr_loss
                else:
                    attr_total_loss_per_batch += attr_loss

            total += data.shape[0]
            for attr in loss.keys():
                total_loss[attr] += loss[attr].item()
                cf_loss[attr] += loss_cf[attr].item()
            ae_loss += loss_ae.item()

            if batch_idx % save_image_interval == 0:
                filepath = "{}/inp_{}.jpg".format(self.epoch, batch_idx)
                self.logger.save_image_batch(data.data, filepath)
                filepath = "{}/out_{}.jpg".format(self.epoch, batch_idx)
                self.logger.save_image_batch(x_prime.data, filepath)
        
        self.logger.log_scalar("test/ae_loss", ae_loss / total, self.epoch)
        
        for attr in pred.keys():
            self.logger.log_scalar("test/{0}loss".format(str(attr)), total_loss[attr] / total, self.epoch)
        
            self.logger.log_scalar("test/{0}_non_priv_pred_loss".format(str(attr)), cf_loss[attr] / total,
                                   self.epoch)
            self.logger.log_scalar("test/{0}_non_priv_accuracy".format(str(attr)), correct[attr] / total,
                                   self.epoch)
            self.logger.log_console("Attribute: {}".format(str(attr)))
            self.logger.log_console("test epoch {}, loss {:.4f}, priv_pred_loss {:.4f}, priv_accuracy {:.3f}"
                                    .format(self.epoch,
                                            total_loss[attr] / total,
                                            cf_loss[attr] / total,
                                            correct[attr] / total))
        self.logger.log_console("ae_loss: {:.3f}".format(ae_loss / total))
                
    def tiprdc_test(self):
        num_logits = self.config["logits"]
        estimator_loss = 0
        classifier_loss = 0

        total_loss, correct, cf_loss, ae_loss = 0, 0, 0, 0
        total_loss = dict()
        correct = dict()
        cf_loss = dict()
        self.classifier.eval()
        for attr in self.classifier.models.keys():
            self.classifier.models[attr].eval()
            total_loss[attr] = 0
            correct[attr] = 0
            cf_loss[attr] = 0

        self.encoder.eval()
        self.mi_estimator.eval()
        total, total_loss, correct, cf_loss, jsd_loss = 0, 0, 0, 0, 0
        for batch_idx, (data, labels) in enumerate(self.testloader):
            data = Variable(data).to(self.device)
            labels = Variable(labels).to(self.device)

            x = data
            x_prime = torch.cat((x[1:], x[0].unsqueeze(0)), dim=0)

            labels_onehot = torch.FloatTensor(data.shape[0], num_logits).to(self.device)
            labels_onehot.zero_()
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)

            z = self.encoder(x)
            pred = self.classifier(z)
            correct += (pred.argmax(dim=1) == labels).sum().item()
            loss_cf = self.loss_fn(pred, labels)
            loss_jsd = -self.mi_loss(self.mi_estimator, x, z, labels_onehot, x_prime)
            loss = -self.lambda_ * loss_cf + (1. - self.lambda_) * loss_jsd

            total += data.shape[0]
            total_loss += loss.item()
            cf_loss += loss_cf.item()
            jsd_loss += loss_jsd.item()

        self.logger.log_scalar("test/loss", total_loss / total, self.epoch)
        self.logger.log_scalar("test/mi_loss", jsd_loss / total, self.epoch)
        self.logger.log_scalar("test/non_priv_pred_loss", cf_loss / total,
                               self.epoch)
        self.logger.log_scalar("test/non_priv_accuracy", correct / total,
                               self.epoch)
        self.logger.log_console("test epoch {}, loss {:.4f}, priv_pred_loss {:.4f}, priv_accuracy {:.3f}, mi_loss {:.3f}"
                                .format(self.epoch,
                                        total_loss / total,
                                        cf_loss / total,
                                        correct / total,
                                        jsd_loss / total))
                

    def ours_test(self):
        self.vae.eval()
        self.aligner.eval()
        self.adv_model.eval()
        save_image_interval = 200
        vae_loss, aligner_loss, adv_loss, dcorr_loss, aligner_correct, adv_correct = 0, 0, 0, 0, 0, 0 
        adv_loss = torch.tensor(0.).to(self.device)
        total = 0
        os.mkdir("{}/{}".format(self.config["log_path"], self.epoch))

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(self.testloader):
                data = Variable(data)
                data = data.to(self.device)
                labels = Variable(labels).to(self.device)
                recon_batch, mu, logvar, z = self.vae(data)
                priv_z, non_priv_z = z[:, :self.priv_dim], z[:, self.priv_dim:]

                total_loss, rec_loss, kld_loss = self.vae_loss_fn(recon_batch, data, mu, logvar, self.beta)
                vae_loss += total_loss.item()
                priv_prediction = self.aligner(priv_z)
                adv_pred = self.adv_model(non_priv_z)

                aligner_loss += self.aligner_loss_fn(priv_prediction, labels)
                adv_loss += self.adv_loss_fn(adv_pred, labels)

                aligner_correct += (priv_prediction.argmax(dim=1) == labels).sum().item()
                adv_correct += (adv_pred.argmax(dim=1) == labels).sum().item()

                if self.dcorr:
                    dcorr_loss += self.dcorr_fn(priv_z, non_priv_z)

                total += int(data.shape[0])

                if batch_idx % save_image_interval == 0:
                    filepath = "{}/inp_{}.jpg".format(self.epoch, batch_idx)
                    self.logger.save_image_batch(data.data, filepath)
                    filepath = "{}/out_{}.jpg".format(self.epoch, batch_idx)
                    self.logger.save_image_batch(recon_batch.data, filepath)

        test_loss = self.alpha1 * vae_loss + self.alpha2 * aligner_loss - self.alpha3 * adv_loss
        if self.dcorr:
            test_loss += self.alpha4 * dcorr_loss
        test_loss /= total
        vae_loss /= total
        aligner_loss /= total
        aligner_acc = aligner_correct / total
        adv_loss /= total
        adv_acc = adv_correct / total
        dcorr_loss /= total

        self.logger.log_scalar("test/loss", test_loss.item(), self.epoch)
        self.logger.log_scalar("test/vae_loss", vae_loss, self.epoch)
        self.logger.log_scalar("test/aligner_loss", aligner_loss.item(), self.epoch)
        self.logger.log_scalar("test/aligner_accuracy", aligner_acc, self.epoch)
        self.logger.log_scalar("test/adv_loss", adv_loss.item(), self.epoch)
        self.logger.log_scalar("test/adv_accuracy", adv_acc, self.epoch)
        self.logger.log_scalar("test/dcorr_loss", dcorr_loss, self.epoch)

        self.logger.log_console("epoch {}, average test loss {:.4f}, vae_loss {:.4f}, aligner_loss {:.4f}, aligner_acc {:.3f}, adv_loss {:.4f}, adv_acc {:.3f}, dcorr_loss {:.3f}".format(self.epoch,
                                                              test_loss,
                                                              vae_loss,
                                                              aligner_loss,
                                                              aligner_acc,
                                                              adv_loss,
                                                              adv_acc,
                                                              dcorr_loss))

    def train(self):
        if self.config["method"] == "ours":
            self.ours_train()
        elif self.config["method"] == "tiprdc":
            self.tiprdc_train()
        elif self.config["method"] == "adversarial":
            self.adversarial_train()
        elif self.config["method"] == "maxentropy":
            self.maxentropy_train() 
        elif self.config["method"] == "gap":
            self.gap_train()
        elif self.config["method"] == "noprivacy":
            self.noprivacy_train()
        elif self.config["method"] == "noise":
            self.noise_train()
        elif self.config["method"] == "vae":
            self.vae_train()

    def vae_train(self):
        # set up training mode
        self.vae.train()
        self.optim.zero_grad()
        
        # initialize scalars
        total, train_loss = 0, 0

        # training loop
        for batch_idx, (data, labels) in enumerate(self.trainloader):
            data = Variable(data).to(self.device)
            labels = Variable(labels).to(self.device)

            # pass through VAE
            recon_batch, mu, logvar, z = self.vae(data)

            loss, rec_loss, kld_loss = self.vae_loss_fn(recon_batch, data, mu, logvar, self.beta)

            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            train_loss += loss.item()
            total += int(data.shape[0])
            step_num = len(self.trainloader) * self.epoch + batch_idx
            self.logger.log_scalar("train/vae_loss", loss.item(), step_num)
            if batch_idx % 100 == 0:
                self.logger.log_console("train epoch {}, iter {}, loss {:.4f}"
                                        .format(self.epoch, batch_idx,
                                                loss / len(data)))
                self.logger.log_model_stats(self.vae)
                self.logger.save_model(self.vae, self.vae_model_path)

        self.logger.log_console("epoch {}, train loss {:.4f}"
                                .format(self.epoch, train_loss / total))
        self.epoch += 1

    def noise_train(self):
        total = 0
        classifier_loss = 0
        self.classifier.train()
        total_loss, correct = 0, 0
        for batch_idx, (data, labels) in enumerate(self.trainloader):
            data = Variable(data).to(self.device)
            labels = Variable(labels).to(self.device)
            self.optim_classifier.zero_grad()

            data = data + torch.randn(data.size()).to(self.device) * self.sigma
            pred = self.classifier(data)
            loss_ce = self.loss_fn(pred, labels)
            loss_ce.backward()
            self.optim_classifier.step()

            correct += (pred.argmax(dim=1) == labels).sum().item()
            
            total += data.shape[0]
            total_loss += loss_ce.item()

            step_num = len(self.trainloader) * self.epoch + batch_idx
            self.logger.log_scalar("train/loss", loss_ce.item(), step_num)
            self.logger.log_scalar("train/adv_accuracy", correct / total,
                                   step_num)
            if batch_idx % 100 == 0:
                self.logger.log_console("train epoch {}, iter {}, loss {:.4f}, priv_accuracy {:.3f}"
                                        .format(self.epoch, batch_idx,
                                                total_loss / total,
                                                correct / total))
        self.logger.log_console("train epoch {}, loss {:.4f}, priv_accuracy {:.3f}"
                                .format(self.epoch,
                                        total_loss / total,
                                        correct / total))
        self.logger.save_model(self.classifier, self.classifier_path)
        self.epoch += 1

    def noprivacy_train(self):
        total = 0
        classifier_loss = 0
        self.classifier.train()
        total_loss, correct = 0, 0
        for batch_idx, (data, labels) in enumerate(self.trainloader):
            data = Variable(data).to(self.device)
            labels = Variable(labels).to(self.device)
            self.optim_classifier.zero_grad()

            pred = self.classifier(data)
            loss_ce = self.loss_fn(pred, labels)
            loss_ce.backward()
            self.optim_classifier.step()

            correct += (pred.argmax(dim=1) == labels).sum().item()
            
            total += data.shape[0]
            total_loss += loss_ce.item()

            step_num = len(self.trainloader) * self.epoch + batch_idx
            self.logger.log_scalar("train/loss", loss_ce.item(), step_num)
            self.logger.log_scalar("train/adv_accuracy", correct / total,
                                   step_num)
            if batch_idx % 100 == 0:
                self.logger.log_console("train epoch {}, iter {}, loss {:.4f}, priv_accuracy {:.3f}"
                                        .format(self.epoch, batch_idx,
                                                total_loss / total,
                                                correct / total))
        self.logger.log_console("train epoch {}, loss {:.4f}, priv_accuracy {:.3f}"
                                .format(self.epoch,
                                        total_loss / total,
                                        correct / total))
        self.logger.save_model(self.classifier, self.classifier_path)
        self.epoch += 1

    def gap_train(self):
        total = 0
        classifier_loss = 0
        self.classifier.train()
        self.decoder.train()
        total_loss, correct, distortion_loss, loss_ce = 0, 0, 0, 0
        ce_loss = 0
        for batch_idx, (data, labels) in enumerate(self.trainloader):
            data = Variable(data).to(self.device)
            labels = Variable(labels).to(self.device)
            self.optim_decoder.zero_grad()
            self.optim_classifier.zero_grad()

            z = torch.randn(data.size(0), 100).to(data.device)
            noise = self.decoder(z)
            x_prime = data + noise
            pred = self.classifier(x_prime)
            loss_ce = self.loss_fn(pred, labels)
            loss_ce.backward()
            self.optim_classifier.step()

            # reset grads for the next pass
            self.optim_classifier.zero_grad()
            self.optim_decoder.zero_grad()


            z = torch.randn(data.size(0), 100).to(data.device)
            noise = self.decoder(z)
            x_prime = data + noise
            pred = self.classifier(x_prime)
            loss_ce = self.loss_fn(pred, labels)
            loss_distortion = torch.max(torch.tensor(0.).to(self.device), self.distortion_loss_fn(x_prime, data) - self.D)

            correct += (pred.argmax(dim=1) == labels).sum().item()
            loss = - self.lambda_ * loss_ce + (1. - self.lambda_) * loss_distortion
            loss.backward()
            self.optim_decoder.step()
            
            total += data.shape[0]
            total_loss += loss.item()
            ce_loss += loss_ce.item()
            distortion_loss += loss_distortion.item()


            step_num = len(self.trainloader) * self.epoch + batch_idx
            self.logger.log_scalar("train/loss", loss.item(), step_num)
            self.logger.log_scalar("train/adv_loss", loss_ce.item(), step_num)
            self.logger.log_scalar("train/distortion_loss", loss_distortion.item(),
                                   step_num)
            self.logger.log_scalar("train/adv_accuracy", correct / total,
                                   step_num)
            if batch_idx % 100 == 0:
                self.logger.log_console("train epoch {}, iter {}, loss {:.4f}, priv_pred_loss {:.4f}, priv_accuracy {:.3f}, distortion_loss {:.3f}"
                                        .format(self.epoch, batch_idx,
                                                total_loss / total,
                                                ce_loss / total,
                                                correct / total,
                                                distortion_loss / total))
        self.logger.log_console("train epoch {}, loss {:.4f}, priv_pred_loss {:.4f}, priv_accuracy {:.3f}, distortion_loss {:.3f}"
                                .format(self.epoch,
                                        total_loss / total,
                                        ce_loss / total,
                                        correct / total,
                                        distortion_loss / total))
        self.logger.save_model(self.classifier, self.classifier_path)
        self.logger.save_model(self.decoder, self.decoder_path)
        self.epoch += 1
    
    def maxentropy_train(self):
        total = 0
        estimator_loss = 0
        classifier_loss = 0
        self.encoder.train()
        self.classifier.train()
        self.decoder.train()
        total_loss, correct, ae_loss, entropy_loss = 0, 0, 0, 0
        for batch_idx, (data, labels) in enumerate(self.trainloader):
            data = data.to(self.device)
            labels = labels.to(self.device)

            z = self.encoder(data)

            pred = self.classifier(z)
            x_prime = self.decoder(z)
            correct += (pred.argmax(dim=1) == labels).sum().item()

            loss_ae = self.reconstruction_loss(data, x_prime)
            loss_entropy = self.entropy_loss(pred)
            loss = -self.lambda_ * loss_entropy + (1. - self.lambda_) * loss_ae

            self.optim_encoder.zero_grad()
            self.optim_classifier.zero_grad()
            loss.backward()
            self.optim_encoder.step()
            self.optim_encoder.zero_grad()

            # Second part of the optimization and hence re-run things
            pred = self.classifier(z.detach())
            loss_cf = self.loss_fn(pred, labels)

            self.optim_classifier.zero_grad()
            loss_cf.backward()
            self.optim_classifier.step()

            total += data.shape[0]
            total_loss += loss.item()
            ae_loss += loss_ae.item()
            entropy_loss += loss_entropy.item()

            step_num = len(self.trainloader) * self.epoch + batch_idx
            self.logger.log_scalar("train/loss", loss.item(), step_num)
            self.logger.log_scalar("train/entropy_loss", loss_entropy.item(), step_num)
            self.logger.log_scalar("train/ae_loss", loss_ae.item(),
                                   step_num)
            self.logger.log_scalar("train/non_priv_accuracy", correct / total,
                                   step_num)
            if batch_idx % 100 == 0:
                self.logger.log_console("train epoch {}, iter {}, loss {:.4f}, priv_pred_loss {:.4f}, priv_accuracy {:.3f}, maxentropy_loss {:.3f}"
                                        .format(self.epoch, batch_idx,
                                                total_loss / total,
                                                ae_loss / total,
                                                correct / total,
                                                entropy_loss / total))
        self.logger.log_console("train epoch {}, loss {:.4f}, priv_pred_loss {:.4f}, priv_accuracy {:.3f}, maxentropy_loss {:.3f}"
                                .format(self.epoch,
                                        total_loss / total,
                                        ae_loss / total,
                                        correct / total,
                                        entropy_loss / total))
        self.logger.save_model(self.encoder, self.encoder_path)
        self.logger.save_model(self.classifier, self.classifier_path)
        self.logger.save_model(self.decoder, self.decoder_path)
        self.epoch += 1

    def adversarial_train(self):
        total = 0
        estimator_loss = 0
        classifier_loss = 0
        self.encoder.train()
        self.classifier.train()
        self.decoder.train()
        total_loss, correct, cf_loss, ae_loss = 0, 0, 0, 0
        total_loss = dict()
        correct = dict()
        cf_loss = dict()
        for attr in self.classifier.models.keys():
            self.classifier.models[attr].train()
            total_loss[attr] = 0
            correct[attr] = 0
            cf_loss[attr] = 0
        for batch_idx, (data, labels) in enumerate(self.trainloader):
            data = data.to(self.device)
            labels = labels.to(self.device)

            z = self.encoder(data)

            pred = self.classifier(z)
            x_prime = self.decoder(z)
            loss_cf = dict()
            for attr in pred.keys():
                
                label_ind = self.config["attribute"].index(attr)
                label = labels[:, label_ind]
                attr_correct = (pred[attr].argmax(dim=1) == label).sum().item()
                correct[attr] += attr_correct
                
                attr_loss_cf = self.loss_fn(pred[attr], label.type(torch.LongTensor))
                loss_cf[attr] = attr_loss_cf

            loss_ae = self.reconstruction_loss(data, x_prime)
            loss = dict()
            attr_total_loss_per_batch = None

            for attr in loss_cf.keys():
                attr_loss = -self.lambda_ * loss_cf[attr] + (1. - self.lambda_) * loss_ae
                loss[attr] = attr_loss
                if attr_total_loss_per_batch is None:
                    attr_total_loss_per_batch = attr_loss
                else:
                    attr_total_loss_per_batch += attr_loss

            self.optim_ae.zero_grad()

            for attr in self.optim_classifier.keys():
                self.optim_classifier[attr].zero_grad()

            attr_total_loss_per_batch.backward()
            self.optim_ae.step()
            self.optim_ae.zero_grad()

            # Second part of the optimization and hence re-run things
            pred = self.classifier(z.detach())
            loss_cf = dict()
            for attr in pred.keys():
                label_ind = self.config["attribute"].index(attr)
                label = labels[:, label_ind]
                attr_loss_cf = self.loss_fn(pred[attr], label.type(torch.LongTensor))
                loss_cf[attr] = attr_loss_cf

            for attr in self.optim_classifier.keys():
                self.optim_classifier[attr].zero_grad()
                loss_cf[attr].backward()
                self.optim_classifier[attr].step()

            total += data.shape[0]
            for attr in loss.keys():
                total_loss[attr] += loss[attr].item()
                cf_loss[attr] += loss_cf[attr].item()
            ae_loss += loss_ae.item()

            step_num = len(self.trainloader) * self.epoch + batch_idx

            self.logger.log_scalar("train/ae_loss", loss_ae.item(), step_num)
            for attr in pred.keys():
                self.logger.log_scalar("train/{0}_loss".format(str(attr)), loss[attr].item(), step_num)
                self.logger.log_scalar("train/{0}_non_priv_pred_loss".format(str(attr)), loss_cf[attr].item(),
                                       step_num)
                self.logger.log_scalar("train/{0}_non_priv_accuracy".format(str(attr)), correct[attr] / total,
                                       step_num)
            if batch_idx % 100 == 0:
                self.logger.log_console("train epoch {}, iter {}, ae_loss: {:.3f}".format(self.epoch, batch_idx, ae_loss / total))
                for attr in pred.keys():
                    self.logger.log_console("Related to: {}".format(attr))
                    self.logger.log_console("loss {:.4f}, priv_pred_loss {:.4f}, priv_accuracy {:.3f}"
                                            .format(total_loss[attr] / total,
                                                    cf_loss[attr] / total,
                                                    correct[attr] / total))
        self.logger.log_console("train epoch {}, iter {}, ae_loss: {:.3f}".format(self.epoch, batch_idx, ae_loss / total))
        for attr in pred.keys():
            self.logger.log_console("Related to: {}".format(attr))
            self.logger.log_console("loss {:.4f}, priv_pred_loss {:.4f}, priv_accuracy {:.3f}"
                                    .format(total_loss[attr] / total,
                                            cf_loss[attr] / total,
                                            correct[attr] / total))
        self.logger.save_model(self.encoder, self.encoder_path)
        self.logger.save_model(self.decoder, self.decoder_path)
        for attr in pred.keys():
            classifier_path = self.classifier_path + str(attr)
            self.logger.save_model(self.classifier, classifier_path)
        self.epoch += 1

    def tiprdc_train(self):
        num_logits = self.config["logits"]
        total = 0
        estimator_loss = 0
        classifier_loss = 0
        self.encoder.train()
        self.classifier.train()
        self.mi_estimator.train()
        total_loss, correct, cf_loss, jsd_loss = 0, 0, 0, 0
        for batch_idx, (data, labels) in enumerate(self.trainloader):
            data = data.to(self.device)
            labels = labels.to(self.device)

            x = data
            x_prime = torch.cat((x[1:], x[0].unsqueeze(0)), dim=0)

            labels_onehot = torch.FloatTensor(data.shape[0], num_logits).to(self.device)
            labels_onehot.zero_()
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)

            z = self.encoder(data)
            pred = self.classifier(z)
            correct += (pred.argmax(dim=1) == labels).sum().item()
            loss_cf = self.loss_fn(pred, labels)
            loss_jsd = -self.mi_loss(self.mi_estimator, x, z, labels_onehot, x_prime)
            loss = -self.lambda_ * loss_cf + (1. - self.lambda_) * loss_jsd

            self.optim_encoder.zero_grad()
            loss.backward()
            self.optim_encoder.step()
            self.optim_encoder.zero_grad()

            # Second part of the optimization and hence re-run things
            z = self.encoder(data)
            z.detach_()
            pred = self.classifier(z)
            loss_cf = self.loss_fn(pred, labels)
            loss_jsd = -self.mi_loss(self.mi_estimator, x, z, labels_onehot, x_prime)

            self.optim_classifier.zero_grad()
            loss_cf.backward(retain_graph=True)
            self.optim_classifier.step()
            pred.detach_()

            self.optim_estimator.zero_grad()
            loss_jsd.backward()
            self.optim_estimator.step()

            total += data.shape[0]
            total_loss += loss.item()
            cf_loss += loss_cf.item()
            jsd_loss += loss_jsd.item()

            step_num = len(self.trainloader) * self.epoch + batch_idx
            self.logger.log_scalar("train/loss", loss.item(), step_num)
            self.logger.log_scalar("train/mi_loss", loss_jsd.item(), step_num)
            self.logger.log_scalar("train/adv_loss", loss_cf.item(),
                                   step_num)
            self.logger.log_scalar("train/non_priv_accuracy", correct / total,
                                   step_num)
            if batch_idx % 100 == 0:
                self.logger.log_console("train epoch {}, iter {}, loss {:.4f}, priv_pred_loss {:.4f}, priv_accuracy {:.3f}, mi_loss {:.3f}"
                                        .format(self.epoch, batch_idx,
                                                total_loss / total,
                                                cf_loss / total,
                                                correct / total,
                                                jsd_loss / total))
        self.logger.log_console("train epoch {}, loss {:.4f}, priv_pred_loss {:.4f}, priv_accuracy {:.3f}, mi_loss {:.3f}"
                                .format(self.epoch,
                                        total_loss / total,
                                        cf_loss / total,
                                        correct / total,
                                        jsd_loss / total))
        self.logger.save_model(self.encoder, self.encoder_path)
        self.logger.save_model(self.classifier, self.classifier_path)
        self.logger.save_model(self.mi_estimator, self.estimator_path)
        self.epoch += 1
                
    def clip_grads(self):
        grad_val = 50.
        torch.nn.utils.clip_grad_norm_(self.vae.module.encoder.parameters(), grad_val)
        torch.nn.utils.clip_grad_norm_(self.vae.module.decoder.parameters(), grad_val)
        #torch.nn.utils.clip_grad_norm_(self.aligner.parameters(), grad_val)
        torch.nn.utils.clip_grad_norm_(self.adv_model.parameters(), grad_val)

    def ours_train(self):
        # set up training mode
        self.vae.train()
        self.aligner.train()
        self.adv_model.train()
        self.optim.zero_grad()
        self.adv_optim.zero_grad()
        
        # initialize scalars
        dcorr_loss, adv_loss = torch.tensor(0.), torch.tensor(0.)
        train_loss, dcorr_loss_total, vae_loss_total = 0, 0, 0
        aligner_correct, adv_correct = 0, 0
        aligner_loss_total, aligner_correct_total, adv_loss_total, adv_correct_total, total = 0, 0, 0, 0, 0
        step_num, warm_start = 1, 0

        # training loop
        for batch_idx, (data, labels) in enumerate(self.trainloader):
            data = Variable(data).to(self.device)
            labels = Variable(labels).to(self.device)

            # pass through global decoupler
            recon_batch, mu, logvar, z = self.vae(data)
            priv_z, non_priv_z = z[:, :self.priv_dim], z[:, self.priv_dim:]

            vae_loss, rec_loss, kld_loss = self.vae_loss_fn(recon_batch, data, mu, logvar, self.beta)

            # pass through aligner
            aligner_pred = self.aligner(priv_z)
            aligner_correct = (aligner_pred.argmax(dim=1) == labels).sum().item()
            aligner_loss = self.aligner_loss_fn(aligner_pred, labels)

            loss = self.alpha1 * vae_loss
            loss += self.alpha2 * aligner_loss
            if self.epoch >= warm_start:
                if self.adv:
                    # pass through adversary
                    adv_pred = self.adv_model(non_priv_z)
                    adv_loss = self.adv_loss_fn(adv_pred, labels)
                    loss += -1 * self.alpha3 * adv_loss

                if self.dcorr:
                    dcorr_loss = self.dcorr_fn(priv_z, non_priv_z)
                    loss += self.alpha4 * dcorr_loss
                    dcorr_loss_total += dcorr_loss.item()

            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            self.adv_optim.zero_grad()

            # adversary optimization
            if self.adv and self.epoch >= warm_start:
                adv_pred = self.adv_model(non_priv_z.detach())
                adv_correct = (adv_pred.argmax(dim=1) == labels).sum().item()
                adv_loss = self.adv_loss_fn(adv_pred, labels)
                adv_loss.backward()
                self.adv_optim.step()
                # optim doesn't need this because we performed detach
                # self.optim.zero_grad()
                self.adv_optim.zero_grad()

            train_loss += loss.item()
            vae_loss_total += vae_loss.item()
            aligner_loss_total += aligner_loss.item()
            aligner_correct_total += aligner_correct
            adv_loss_total += adv_loss.item()
            adv_correct_total += adv_correct
            total += int(data.shape[0])
            step_num = len(self.trainloader) * self.epoch + batch_idx
            self.logger.log_scalar("train/loss", loss.item(), step_num)
            self.logger.log_scalar("train/vae_loss", vae_loss.item(), step_num)
            self.logger.log_scalar("train/rec_loss", rec_loss.item(), step_num)
            self.logger.log_scalar("train/kld_loss", kld_loss.item(), step_num)
            self.logger.log_scalar("train/aligner_loss", aligner_loss.item(),
                                   step_num)
            self.logger.log_scalar("train/adv_loss", adv_loss.item(),
                                   step_num)
            self.logger.log_scalar("train/aligner_accuracy", aligner_correct / total,
                                   step_num)
            self.logger.log_scalar("train/adv_accuracy", adv_correct / total,
                                   step_num)
            self.logger.log_scalar("train/dcorr_loss", dcorr_loss_total / total, step_num)
            if batch_idx % 100 == 0:
                self.logger.log_console("train epoch {}, iter {}, loss {:.4f}, vae_loss {:.4f}, aligner_loss {:.4f}, aligner_accuracy {:.3f}, adv_loss {:.4f}, adv_accuracy {:.3f}, dcorr_loss {:.3f}"
                                        .format(self.epoch, batch_idx,
                                                loss / len(data),
                                                vae_loss / len(data),
                                                aligner_loss / len(data),
                                                aligner_correct / len(data),
                                                adv_loss / len(data),
                                                adv_correct / len(data),
                                                dcorr_loss / len(data)))
                self.logger.log_model_stats(self.vae)
                self.logger.log_model_stats(self.aligner)
                self.logger.log_model_stats(self.adv)
                self.logger.save_model(self.vae, self.vae_model_path)
                self.logger.save_model(self.aligner, self.aligner_model_path)
                self.logger.save_model(self.adv_model, self.adv_model_path)

        self.logger.log_console("epoch {}, train loss {:.4f}, vae_loss {:.4f}, aligner_loss {:.4f}, aligner_accuracy {:.3f}, adv_loss {:.4f}, adv_accuracy {:.4f}, dcorr_loss {:.3f}"
                                .format(self.epoch, train_loss / total,
                                        vae_loss_total / total,
                                        aligner_loss_total / total,
                                        aligner_correct_total / total,
                                        adv_loss_total / total,
                                        adv_correct_total / total,
                                        dcorr_loss_total / total))
        self.epoch += 1
