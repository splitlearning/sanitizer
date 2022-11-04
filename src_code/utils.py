import os
from shutil import copytree, copy2
from glob import glob
import torchvision
import torch
from tensorboardX import SummaryWriter


def copy_source_code(path):
    if not os.path.isdir(path):
        os.makedirs(path)

    denylist = ["./__pycache__/"]
    folders = glob(r'./*/')

    # For copying python files
    for file_ in glob(r'./*.py'):
        copy2(file_, path)

    for file_ in glob(r'./*.json'):
        copy2(file_, path)


    for folder in folders:
        if folder not in denylist:
            # Remove first char which is . due to the glob
            copytree(folder, path + folder[1:])


class LoggerUtils():
    """docstring for LoggerUtils"""

    def __init__(self, config):
        super(LoggerUtils, self).__init__()
        log_path = config["log_path"]
        tb_path = log_path + "/tensorboard"
        os.mkdir(tb_path)
        self.config = config
        self.writer = SummaryWriter(tb_path)

    def save_image_batch(self, batch_image, path):
        torchvision.utils.save_image(batch_image, '{}/{}'.format(
            self.config["log_path"], path), nrow=8, padding=2)

    def save_model(self, model, path):
        torch.save(model.state_dict(), path)

    def log_model_stats(self, model):
        pass

    def log_console(self, message):
        print(message)

    def log_scalar(self, category, value, iteration):
        self.writer.add_scalar(category, value, iteration)
