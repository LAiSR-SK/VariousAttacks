# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
import ssl

import torch
import torchvision
from torchvision import transforms

from va.data.cifar import CIFAR100


def load_data(ds_name, args, kwargs, coarse=False):
    """
    Loads the specified dataset into a dataloader and returns the data split
    into training and test loaders

    :param ds_name: name of the dataset to load for training
    :param args: program arguments
    :param kwargs: more program arguments

    :return data_loaders containing the training and testing sets
    """

    # Establish the data loader transforms
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    ssl._create_default_https_context = (
        ssl._create_unverified_context
    )  # set the context for working with tensors

    if ds_name == "cifar10":
        # Load in the CIFAR10 dataloaders
        trainset = torchvision.datasets.CIFAR10(
            root="data/download",
            train=True,
            download=True,
            transform=transform_train,
        )
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, **kwargs
        )
        testset = torchvision.datasets.CIFAR10(
            root="data/download",
            train=False,
            download=True,
            transform=transform_test,
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False, **kwargs
        )
    elif ds_name == "cifar100":
        # Load in the CIFAR100 dataloaders
        if coarse:
            trainset = CIFAR100(
                root="data/download",
                train=True,
                download=True,
                transform=transform_train,
                coarse=True,
            )
            train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=args.batch_size, shuffle=True, **kwargs
            )
            testset = CIFAR100(
                root="data/download",
                train=False,
                download=True,
                transform=transform_test,
                coarse=True,
            )
            test_loader = torch.utils.data.DataLoader(
                testset,
                batch_size=args.test_batch_size,
                shuffle=False,
                **kwargs,
            )
        else:
            trainset = torchvision.datasets.CIFAR100(
                root="data/download",
                train=True,
                download=True,
                transform=transform_train,
            )
            train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=args.batch_size, shuffle=True, **kwargs
            )
            testset = torchvision.datasets.CIFAR100(
                root="data/download",
                train=False,
                download=True,
                transform=transform_test,
            )
            test_loader = torch.utils.data.DataLoader(
                testset,
                batch_size=args.test_batch_size,
                shuffle=False,
                **kwargs,
            )
    else:
        raise NotImplementedError

    return train_loader, test_loader
