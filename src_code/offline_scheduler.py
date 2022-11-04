import torch
from torchvision import datasets, transforms
from torch.distributions.categorical import Categorical
import dataset_utils
import numpy as np
from torch.autograd import Variable
import os
from utils import LoggerUtils
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from torch.distributions.laplace import Laplace
from ron_gauss import RONGauss


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

        if self.config["dataset"] == "CelebA":
            train_config["train"] = True
            train_dataset = dataset_utils.CelebA(train_config)
            train_config["train"] = False
            test_dataset = dataset_utils.CelebA(train_config)
        elif self.config["dataset"] == "UTKFace":
            train_config["format"] = "jpg"
            dataset = dataset_utils.UTKFace(train_config)
        elif self.config["dataset"] == "FairFace":
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

        self.testloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.config["test_batch_size"],
            shuffle=False, num_workers=5)

    def load_mean_z(self):
        # dirty implementation to get trainloader
        sens_attrib = "gender"
        trainTransform = transforms.Compose([
            transforms.Resize((self.IM_SIZE, self.IM_SIZE)),
            transforms.ToTensor()])

        train_config = {"transforms": trainTransform,
                        "path": self.config["dataset_path"],
                        "attribute": sens_attrib}
        train_config["format"] = "jpg"
        dataset = dataset_utils.UTKFace(train_config)
        train_dataset, test_dataset = self.get_split(dataset)

        sens_trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config["train_batch_size"],
            shuffle=False, num_workers=5)

        z_0 = torch.empty((0, self.nz), dtype=torch.float32, device=self.device)
        z_1 = torch.empty((0, self.nz), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            #TODO: poor implementation for getting mean, implement running mean
            for data, label in sens_trainloader:
                img = data.to(self.device)
                z, _ = self.vae.module.encode(img)
                idx_0 = (0 == label).nonzero(as_tuple=True)[0]
                idx_1 = (1 == label).nonzero(as_tuple=True)[0]
                z_0 = torch.cat((z_0, z[idx_0]), axis=0)
                z_1 = torch.cat((z_1, z[idx_1]), axis=0)
        self.z_class_0 = z_0.mean(dim=0)
        self.z_class_1 = z_1.mean(dim=0)
        print("obtained mean vectors")

    def load_vae_model(self):
        utility_eval = True
        eval_attribute = "race"#"high_cheekbones"
        self.sens_attribute = "gender"
        wt_path = self.config["model_path"]
        if self.is_cas:
            wt_path = wt_path.replace("_cas", "")
        if utility_eval:
            wt_path = wt_path.replace(eval_attribute, self.sens_attribute)
        if self.config["method"] == "ours":
            wt_path = wt_path.replace(self.mechanism, "")
            if self.mechanism in ["dpgmm", "obfuscation"]:
                wt_path = wt_path.replace("_eps" + str(self.objects["eps"]), "")
        k = wt_path
        wt_path = wt_path.replace("_offline", "") + "/model_vae.pt"
        self.vae.load_state_dict(torch.load(wt_path, map_location=self.device))
        # wt_path = k.replace("_offline", "") + "/model_priv_pred.pt"
        # self.priv_pred.load_state_dict(torch.load(wt_path, map_location=self.device))

    def load_encoder(self):
        utility_eval = True
        eval_attribute = "high_cheekbones"
        sens_attribute = "gender"
        wt_path = self.config["model_path"]
        if self.is_cas:
            wt_path = wt_path.replace("_cas", "")
        if utility_eval:
            wt_path = wt_path.replace(eval_attribute, sens_attribute)
        wt_path = wt_path.replace("_offline", "") + "/encoder.pt"
        self.encoder.load_state_dict(torch.load(wt_path, map_location=self.device))

    def load_decoder(self):
        utility_eval = True
        eval_attribute = "high_cheekbones"
        #eval_attribute = "gender"
        #sens_attribute = "race"
        sens_attribute = "gender"
        wt_path = self.config["model_path"]
        if self.is_cas:
            wt_path = wt_path.replace("_cas", "")
        if utility_eval:
            wt_path = wt_path.replace(eval_attribute, sens_attribute)
        wt_path = wt_path.replace("_offline", "") + "/decoder.pt"
        self.decoder.load_state_dict(torch.load(wt_path, map_location=self.device))

    def ours_learn_dpgmm(self):
        dim, prng_seed = 4, 14 # put them in the config file
        self.dp_gen = RONGauss(algorithm='gmm', epsilon_mean=0.3*self.eps, epsilon_cov=0.7*self.eps)
        X, Y = np.random.rand(0, self.priv_dim), np.random.rand(0)
        total_samples = 20000
        self.unique_label_wt = []

        # dirty implementation to get trainloader
        trainTransform = transforms.Compose([
            transforms.Resize((self.IM_SIZE, self.IM_SIZE)),
            transforms.ToTensor()])

        train_config = {"transforms": trainTransform,
                        "path": self.config["dataset_path"],
                        "attribute": self.sens_attribute,
                        "train": True}
        if self.config["dataset"] == "CelebA":
            train_dataset = dataset_utils.CelebA(train_config)
        elif self.config["dataset"] == "UTKFace":
            dataset = dataset_utils.UTKFace(train_config)
            train_dataset, test_dataset = self.get_split(dataset)
        elif self.config["dataset"] == "FairFace":
            train_dataset = dataset_utils.FairFace(train_config)
        else:
            print("Unknown dataset", self.config["dataset"])
            exit()

        sens_trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config["train_batch_size"],
            shuffle=False, num_workers=5)

        for data, label in sens_trainloader:
            with torch.no_grad():
                z, _ = self.vae.module.encode(data.to(self.device))
                z = z[:, :self.priv_dim].cpu()
            X = np.concatenate([X, z])
            Y = np.concatenate([Y, label])
            if X.shape[0] > total_samples:
                break
        print("data loaded, now fitting dpgmm")
        self.dp_gen.learn_gmm_rongauss(X, dim, Y, prng_seed)
        for label in self.dp_gen.labels:
            self.unique_label_wt.append((label == Y).sum() / Y.shape[0])
        print("gmm fitting done, unique label wt is {}".format(self.unique_label_wt))
        
    def ours_learn_gmm(self):
        # create a big batch
        self.gmm = GaussianMixture(n_components=self.components, random_state=0)
        X = np.random.rand(0, self.priv_dim)
        total_samples = 5000
        for data, _ in self.trainloader:
            with torch.no_grad():
                recon_batch, mu, logvar, z = self.vae(data.cuda())
                z = z[:, :self.priv_dim].cpu()
            X = np.concatenate([X, z])
            if X.shape[0] > total_samples:
                break
        self.gmm.fit(X)
        print("gmm learning over")

    def ours_learn_clusters(self):
        X = np.random.rand(0, self.priv_dim)
        total_samples = 5000
        for data, _ in self.trainloader:
            with torch.no_grad():
                recon_batch, mu, logvar, z = self.vae(data.cuda())
                z = z[:, :self.priv_dim].cpu()
            X = np.concatenate([X, z])
            if X.shape[0] > total_samples:
                break
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=50).fit(X)
        print("clustering over")

    def ours_training_params(self):
        self.epoch = 0
        self.device = self.objects["device"]
        self.priv_dim = self.objects["priv_dim"]
        self.mechanism = self.objects["mechanism"]
        self.vae = self.objects["vae"]
        #self.priv_pred = self.objects["priv_pred"]
        self.load_vae_model()
        self.model = self.objects["model"]
        self.loss_fn = self.objects["pred_loss_fn"]
        # Performing joint optimization right now
        self.optim = self.objects["model_optim"]

        self.model_path = self.config["model_path"] + "/model.pt"
        self.vae.eval()

        if self.mechanism == "sampling":
            self.components = self.objects["components"]
            self.ours_learn_gmm()
        elif self.mechanism == "dpgmm":
            self.eps = self.objects["eps"]
            self.components = self.objects["components"]
            self.ours_learn_dpgmm()
        elif self.mechanism == "obfuscation":
            self.max = 5
            sensitivity = self.max * 2 * self.priv_dim
            eps = self.objects["eps"]
            self.var = sensitivity / eps
        elif self.mechanism == "generalization":
            self.num_clusters = self.objects["clusters"]
            self.ours_learn_clusters()

    def vae_training_params(self):
        self.epoch = 0
        self.device = self.objects["device"]
        self.vae = self.objects["vae"]
        self.nz = self.objects["nz"]
        self.load_vae_model()
        self.load_mean_z()
        self.model = self.objects["model"]
        self.loss_fn = self.objects["pred_loss_fn"]
        # Performing joint optimization right now
        self.optim = self.objects["model_optim"]
        self.model_path = self.config["model_path"] + "/model.pt"
        self.vae.eval()

    def tiprdc_training_params(self):
        self.epoch = 0
        self.device = self.objects["device"]
        self.encoder = self.objects["encoder"]
        self.load_encoder()
        if self.is_cas:
            self.decoder = self.objects["decoder"]
            self.load_decoder()
            self.decoder.eval()
        self.model = self.objects["classifier"]

        self.optim = self.objects["optim_classifier"]

        self.loss_fn = self.objects["loss_fn"]

        self.model_path = self.config["model_path"] + "/model.pt"
        self.encoder.eval()

    def gap_training_params(self):
        self.epoch = 0
        self.device = self.objects["device"]
        self.decoder = self.objects["decoder"]
        self.load_decoder()
        self.model = self.objects["classifier"]

        self.optim = self.objects["optim_classifier"]

        self.loss_fn = self.objects["loss_fn"]

        self.model_path = self.config["model_path"] + "/model.pt"

    def noise_training_params(self):
        self.epoch = 0
        self.device = self.objects["device"]
        self.model = self.objects["classifier"]
        self.sigma = self.objects["sigma"]
        self.optim = self.objects["optim_classifier"]
        self.loss_fn = self.objects["loss_fn"]
        self.model_path = self.config["model_path"] + "/model.pt"

    def setup_training_params(self):
        self.is_cas = self.config["is_cas"]
        if self.config["method"] == "ours":
            self.ours_training_params()
        elif self.config["method"] in ["adversarial", "tiprdc", "maxentropy"]:
            self.tiprdc_training_params()
        elif self.config["method"] == "gap":
            self.gap_training_params()
        elif self.config["method"] == "noise":
            self.noise_training_params()
        elif self.config["method"] == "vae":
            self.vae_training_params()
        else:
            print("method not implemented")
            exit()


    def test(self):
        self.model.eval()
        save_image_interval = 200
        vae_loss, priv_pred_loss, pred_loss, test_loss, correct, non_priv_pred_correct = 0, 0, 0, 0, 0, 0
        total = 0
        os.mkdir("{}/{}".format(self.config["log_path"], self.epoch))

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(self.testloader):
                data = Variable(data)
                data = data.to(self.device)
                labels = Variable(labels).to(self.device)
                if self.is_cas:
                    non_priv_z = data
                else:
                    if self.config["method"] == "ours":
                        recon_batch, mu, logvar, z = self.vae(data)
                        if self.mechanism == "suppression":
                            priv_z, non_priv_z = z[:, :self.priv_dim], z[:, self.priv_dim:]
                        elif self.mechanism == "suppression_gen":
                            z[:, :self.priv_dim] = torch.zeros_like(z[:, :self.priv_dim])
                            non_priv_z = self.vae.module.decode(z)
                        elif self.mechanism == "sampling":
                            priv_features, y_tilde = self.gmm.sample(z.shape[0])
                            np.random.shuffle(priv_features)
                            z[:, :self.priv_dim] = torch.tensor(priv_features).to(self.device)
                            non_priv_z = self.vae.module.decode(z)
                        elif self.mechanism == "obfuscation":
                            # truncate the z_s
                            z[:, :self.priv_dim] = torch.clamp(z[:, :self.priv_dim], -self.max, self.max)
                            # add noise
                            lap_noise = Laplace(torch.zeros(data.shape[0], self.priv_dim), torch.empty(data.shape[0], self.priv_dim).fill_(self.var))
                            z[:, :self.priv_dim] += lap_noise.sample().to(self.device)
                            non_priv_z = self.vae.module.decode(z)
                        elif self.mechanism == "dpgmm":
                            n_samples = []
                            for idx in range(len(self.unique_label_wt) - 1):
                                n_samples.append(int(self.unique_label_wt[idx] * data.shape[0]))
                            n_samples.append(data.shape[0] - sum(n_samples))
                            priv_features, y_tilde = self.dp_gen.sample_dpgmm(n_samples)
                            z[:, :self.priv_dim] = torch.tensor(priv_features).to(self.device)
                            non_priv_z = self.vae.module.decode(z)
                        if self.mechanism in ["sampling", "obfuscation", "suppression_gen", "generalization", "dpgmm"]:
                            if batch_idx % save_image_interval == 0:
                                filepath = "{}/inp_{}.jpg".format(self.epoch, batch_idx)
                                self.logger.save_image_batch(data.data, filepath)
                                filepath = "{}/out_{}.jpg".format(self.epoch, batch_idx)
                                self.logger.save_image_batch(non_priv_z.data, filepath)
                            # if self.mode == "cas":
                                #labels = y_tilde
                    elif self.config["method"] == "vae":
                        batch_size = data.shape[0]
                        dist = Categorical(probs = torch.tensor([0.5, 0, 0.5]).repeat(batch_size, 1))
                        recon_batch, mu, logvar, z = self.vae(data)
                        weights = dist.sample().unsqueeze(1).to(self.device)
                        non_priv_z = z + weights * (self.z_class_1.repeat(batch_size, 1)) +\
                                     (-1 * weights) * (self.z_class_0.repeat(batch_size, 1))
                        rec = self.vae.module.decode(non_priv_z)
                        if batch_idx % save_image_interval == 0:
                                filepath = "{}/inp_{}.jpg".format(self.epoch, batch_idx)
                                self.logger.save_image_batch(data.data, filepath)
                                filepath = "{}/out_{}.jpg".format(self.epoch, batch_idx)
                                self.logger.save_image_batch(rec.data, filepath)
                        if self.is_cas:
                            non_priv_z = self.vae.module.decode(non_priv_z)
                    elif self.config["method"] in ["adversarial", "tiprdc", "maxentropy"]:
                        non_priv_z = self.encoder(data)
                        if self.is_cas:
                            non_priv_z = self.decoder(non_priv_z)
                    elif self.config["method"] == "gap":
                        z = torch.randn(data.size(0), 100).to(data.device)
                        noise = self.decoder(z)
                        non_priv_z = data + noise
                    elif self.config["method"] == "noise":
                        noise = torch.randn(data.size()).to(data.device) * self.sigma
                        non_priv_z = data + noise
                    else:
                        print("unknown method")
                        exit()
                prediction = self.model(non_priv_z)

                pred_loss += self.loss_fn(prediction, labels)
                correct += (prediction.argmax(dim=1) == labels).sum().item()

                total += int(data.shape[0])


        pred_loss /= total
        pred_acc = correct / total

        self.logger.log_scalar("test/pred_loss", pred_loss.item(), self.epoch)
        self.logger.log_scalar("test/pred_accuracy", pred_acc, self.epoch)

        self.logger.log_console("epoch {}, test loss {:.4f}, pred_acc {:.3f}".format(self.epoch,
                                                              pred_loss,
                                                              pred_acc))

    def train(self):
        self.model.train()
        if self.config["method"] == "gap":
            self.decoder.eval()
        elif self.config["method"] in ["adversarial", "tiprdc", "maxentropy"]:
            self.encoder.eval()
        train_loss = 0
        correct_total = 0
        total = 0
        for batch_idx, (data, labels) in enumerate(self.trainloader):
            data = Variable(data).to(self.device)
            labels = Variable(labels).to(self.device)
            self.optim.zero_grad()

            with torch.no_grad():
                if self.config["method"] == "ours":
                    recon_batch, mu, logvar, z = self.vae(data)
                    if self.mechanism == "suppression":
                        priv_z, non_priv_z = z[:, :self.priv_dim], z[:, self.priv_dim:]
                    elif self.mechanism == "suppression_gen":
                            z[:, :self.priv_dim] = torch.zeros_like(z[:, :self.priv_dim])
                            non_priv_z = self.vae.module.decode(z)
                    elif self.mechanism == "sampling":
                        priv_features, y_tilde = self.gmm.sample(z.shape[0])
                        np.random.shuffle(priv_features)
                        z[:, :self.priv_dim] = torch.tensor(priv_features).to(self.device)
                        non_priv_z = self.vae.module.decode(z)
                        # let's assume labels are known for now
                        # if self.mode == "cas":
                    elif self.mechanism == "dpgmm":
                        n_samples = []
                        for idx in range(len(self.unique_label_wt) - 1):
                            n_samples.append(int(self.unique_label_wt[idx] * data.shape[0]))
                        n_samples.append(data.shape[0] - sum(n_samples))
                        priv_features, y_tilde = self.dp_gen.sample_dpgmm(n_samples)
                        z[:, :self.priv_dim] = torch.tensor(priv_features).to(self.device)
                        non_priv_z = self.vae.module.decode(z)
                    elif self.mechanism == "obfuscation":
                        # truncate the z_s
                        z[:, :self.priv_dim] = torch.clamp(z[:, :self.priv_dim], -self.max, self.max)
                        # add noise
                        lap_noise = Laplace(torch.zeros(data.shape[0], self.priv_dim), torch.empty(data.shape[0], self.priv_dim).fill_(self.var))
                        z[:, :self.priv_dim] += lap_noise.sample().to(self.device)
                        non_priv_z = self.vae.module.decode(z)
                    elif self.mechanism == "generalization":
                        nearest_cluster_index = self.kmeans.predict(z[:, :self.priv_dim].cpu().numpy().astype(float))
                        nearest_centroids = self.kmeans.cluster_centers_[nearest_cluster_index]
                        nearest_centroids = torch.tensor(nearest_centroids, dtype=torch.double).to(self.device)
                        z[:, :self.priv_dim] = nearest_centroids
                        non_priv_z = self.vae.module.decode(z)
                    if self.is_cas:
                        labels = torch.argmax(self.priv_pred(z[:, :self.priv_dim]), dim=1)

                elif self.config["method"] == "vae":
                    batch_size = data.shape[0]
                    dist = Categorical(probs = torch.tensor([0.5, 0, 0.5]).repeat(batch_size, 1))
                    recon_batch, mu, logvar, z = self.vae(data)
                    weights = dist.sample().unsqueeze(1).to(self.device)
                    non_priv_z = z + weights * (self.z_class_1.repeat(batch_size, 1)) +\
                                 (-1 * weights) * (self.z_class_0.repeat(batch_size, 1))
                    if self.is_cas:
                        non_priv_z = self.vae.module.decode(non_priv_z)
                elif self.config["method"] in ["adversarial", "tiprdc", "maxentropy"]:
                    non_priv_z = self.encoder(data)
                    if self.is_cas:
                        non_priv_z = self.decoder(non_priv_z)
                elif self.config["method"] == "gap":
                    z = torch.randn(data.size(0), 100).to(data.device)
                    noise = self.decoder(z)
                    non_priv_z = data + noise
                elif self.config["method"] == "noise":
                    noise = torch.randn(data.size()).to(data.device) * self.sigma
                    non_priv_z = data + noise
                else:
                    print("unknown method")
                    exit()

            prediction = self.model(non_priv_z)

            try:
                pred_loss = self.loss_fn(prediction, labels)
            except:
                print(prediction)
                print(labels)
                print(prediction.shape)
                print(labels.shape)
                print(torch.unique(prediction))
                print(torch.unique(labels))
            correct_total += (prediction.argmax(dim=1) == labels).sum().item()

            pred_loss.backward()
            self.optim.step()

            train_loss += pred_loss.item()
            total += int(data.shape[0])

            step_num = len(self.trainloader) * self.epoch + batch_idx
            self.logger.log_scalar("train/pred_loss", pred_loss.item(), step_num)
            self.logger.log_scalar("train/pred_accuracy", correct_total / total,
                                   step_num)
            if batch_idx % 100 == 0:
                self.logger.log_console("train epoch {}, iter {}, loss {:.4f}, pred_accuracy {:.3f}"
                                        .format(self.epoch, batch_idx,
                                                train_loss / total,
                                                correct_total / total))
                self.logger.save_model(self.model, self.model_path)

        self.logger.log_console("epoch {}, train loss {:.4f}, accuracy {:.4f}"
                                .format(self.epoch, train_loss / total,
                                        correct_total / total))
        self.epoch += 1
