import torch.utils.data as data
from torchvision import datasets
from PIL import Image
from glob import glob
from abc import abstractmethod
from copy import deepcopy
import torch
import pandas as pd
import re
import numpy as np


class BaseDataset(data.Dataset):
    """docstring for BaseDataset"""

    def __init__(self, config):
        super(BaseDataset, self).__init__()
        self.format = config["format"]
        self.set_filepaths(config["path"])
        self.transforms = config["transforms"]

    def set_filepaths(self, path):
        filepaths = path + "/*.{}".format(self.format)
        self.filepaths = glob(filepaths)

    def load_image(self, filepath):
        img = Image.open(filepath)
        # img = np.array(img)
        return img

    @staticmethod
    def to_tensor(obj):
        return torch.tensor(obj)

    @abstractmethod
    def load_label(self):
        pass

    def __getitem__(self, index):
        filepath = self.filepaths[index]
        img = self.load_image(filepath)
        img = self.transforms(img)
        label = self.load_label(filepath)
        label = self.to_tensor(label)
        return img, label

    def __len__(self):
        return len(self.filepaths)

class FairFace(BaseDataset):
    """docstring for FairFace"""

    def __init__(self, config):
        config = deepcopy(config)
        self.attribute = config["attribute"]
        if config["train"] is True:
            label_csv = pd.read_csv(config["path"] +
                                    "fairface_label_train.csv")
            config["path"] += "/train"
        else:
            label_csv = pd.read_csv(config["path"] + "fairface_label_val.csv")
            config["path"] += "/val"
        self.label_csv = label_csv.set_index("file")
        super(FairFace, self).__init__(config)
        self.attribute = config["attribute"]
        if self.attribute == "race":
            self.label_mapping = {"East Asian": 0,
                                  "Indian": 1,
                                  "Black": 2,
                                  "White": 3,
                                  "Middle Eastern": 4,
                                  "Latino_Hispanic": 5,
                                  "Southeast Asian": 6}
        elif self.attribute == "age":
            self.label_mapping = {"0-2": 0,
                                  "3-9": 1,
                                  "10-19": 2,
                                  "20-29": 3,
                                  "30-39": 4,
                                  "40-49": 5,
                                  "50-59": 6,
                                  "60-69": 7,
                                  "more than 70": 8}
        elif self.attribute == "gender":
            self.label_mapping = {"Male": 0, "Female": 1}

    def load_label(self, filepath):
        reg_exp = r'//(.*/\d+\.{})'
        filename = re.search(reg_exp.format(self.format), filepath).group(1)
        labels_row = self.label_csv.loc[filename]
        label = labels_row[self.attribute]
        return self.label_mapping[label]
        # filepath = filepath.split("/")


class UTKFace(BaseDataset):
    """docstring for UTKFace"""

    def __init__(self, config):
        super(UTKFace, self).__init__(config)
        self.attribute = config["attribute"]

    def load_label(self, filepath):
        labels = filepath.split("/")[-1].split("_")
        if self.attribute == "race":
            try:
                label = int(labels[2])
            except:
                print("corrupt label")
                label = np.random.randint(0, 4)
        elif self.attribute == "gender":
            label = int(labels[1])
        elif self.attribute == "age":
            # label = float(labels[0])
            if int(labels[0]) < 3:
                label = 0
            elif int(labels[0]) < 10:
                label = 1
            elif int(labels[0]) < 20:
                label = 2
            elif int(labels[0]) < 30:
                label = 3
            elif int(labels[0]) < 40:
                label = 4
            elif int(labels[0]) < 50:
                label = 5
            elif int(labels[0]) < 60:
                label = 6
            elif int(labels[0]) < 70:
                label = 7
            else:
                label = 8
            label = float(label)
        return label


class CelebA(datasets.CelebA):
    def __init__(self, config):
        config = deepcopy(config)
        data_split = "train" if config["train"] else "valid" 
        self.attribute = config["attribute"]
        self.attr_indices = {'gender': 20, 
                             'eyeglasses': 15,
                             'necklace': 37,
                             'smiling': 31,
                             'straight_hair': 32,
                             'wavy_hair': 33,
                             'big_nose': 7,
                             'mouth_open': 21,
                             'high_cheekbones': 19}

        target_pred = 'attr' # remove
        # if self.attribute in self.attr_indices.keys():
        #     target_pred = 'attr'
        # else:
        #     raise ValueError("Prediction Attribute {} is not supported.".format(self.attribute))
            
        super().__init__(root=config["path"], split=data_split, 
                       target_type=target_pred, transform=config["transforms"],
                       download=False)
        
    def __getitem__(self, index):
        img, pred_label = super().__getitem__(index)
        attr_index = self.attr_indices[self.attribute]
        specific_pred_label = 1 if pred_label[attr_index] > 0 else 0
        return img, specific_pred_label
