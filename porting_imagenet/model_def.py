"""

Model definition file template

"""
from typing import Any, Dict, Sequence, Tuple, Union, cast
import os, sys

from determined.pytorch import DataLoader, PyTorchTrial, LRScheduler, PyTorchTrialContext

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class ImagenetTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext):
        # TODO: Create and wrap model and optimizer
        pass 

    def build_training_data_loader(self):
        # TODO: Update download_directory and return a Determined Dataloader

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        trainset = datasets.CIFAR10(
            root=self.download_directory, train=True, download=True, transform=transform
        )
        

        return 

    def build_validation_data_loader(self):
        # TODO: return a Determined Dataloader

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        val_dataset = datasets.CIFAR10(
            root=self.download_directory, train=False, download=True, transform=transform
        )

        return

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int):
        # TODO: Feed batch to model and backprop. Return dictionary of metrics
        
        return {}

    def evaluate_batch(self, batch: TorchData):
        # TODO: Feed batch to model. Return dictionary of metrics

        return {}
