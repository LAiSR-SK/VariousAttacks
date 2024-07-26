# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
import torch
from torch import nn, optim
from torch.autograd import Variable

from va.loss import CWLoss


def cw_whitebox_eval(
    model,
    dataset,
    X,
    y,
    device,
    epsilon=0.031,
    num_steps=20,
    step_size=2.0 / 255.0,
):
    """
    Evaluates the model by perturbing an image using the CW attack.

    :param model: model being attacked
    :param dataset: name of dataset (cifar10 or cifar100)
    :param X: image being attacked
    :param y: correct label of the image being attacked
    :param device: current device
    :param epsilon: epsilon size for the CW attack
    :param num_steps: number of perturbation steps
    :param step_size: step size of perturbation steps

    :return clean error and CW error
    """

    # Calculates clean error of image classification
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()

    # Create X_cw basis by duplicating X as a variable
    X_cw = Variable(X.data, requires_grad=True)

    # If specified, create random noice between - and + epsilon and add to X_cw
    random_noise = (
        torch.FloatTensor(*X_cw.shape).uniform_(-epsilon, epsilon).to(device)
    )
    X_cw = Variable(X_cw.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        # Create the SGD optimizer and zero the gradients
        opt = optim.SGD([X_cw], lr=1e-3)
        opt.zero_grad()

        # With gradients, set up the CW loss and step backward
        with torch.enable_grad():
            loss = CWLoss(100 if dataset == "cifar100" else 10)(model(X_cw), y)
        loss.backward()

        # Calculate the perturbation using the CW loss gradient
        eta = step_size * X_cw.grad.data.sign()
        X_cw = Variable(X_cw.data + eta, requires_grad=True)
        eta = torch.clamp(X_cw.data - X.data, -epsilon, epsilon)

        # Add the perturbation and clamp x_cw
        X_cw = Variable(X.data + eta, requires_grad=True)
        X_cw = Variable(torch.clamp(X_cw, 0, 1.0), requires_grad=True)

    # Calculates the CW error between the prediction and correct classification
    err_cw = (model(X_cw).data.max(1)[1] != y.data).float().sum()

    return err, err_cw


def mim_whitebox_eval(
    model,
    X,
    y,
    device,
    epsilon=0.031,
    num_steps=20,
    step_size=0.031,
    decay_factor=1.0,
):
    """
    Evaluates the model by perturbing an image using the MIM attack.

    :param model: model being attacked
    :param X: image being attacked
    :param y: correct label of the image being attacked
    :param device: current device
    :param epsilon: epsilon size for the MIM attack
    :param num_steps: number of perturbation steps
    :param step_size: step size of perturbation steps
    :param decay_factor: coefficient for previous gradient usage in MIM attack

    :return clean error and MIM error
    """

    # Calculates clean error of image classification
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()

    # Create X_mim basis by duplicating X as a variable
    X_mim = Variable(X.data, requires_grad=True)

    # create random noise between - and + epsilon and add to X_cw
    random_noise = (
        torch.FloatTensor(*X_mim.shape).uniform_(-epsilon, epsilon).to(device)
    )
    X_mim = Variable(X_mim.data + random_noise, requires_grad=True)

    # Set up tensor to hold previous gradients
    previous_grad = torch.zeros_like(X.data)

    for _ in range(num_steps):
        # Create the SGD optimizer and zero the gradients
        opt = optim.SGD([X_mim], lr=1e-3)
        opt.zero_grad()

        # With gradients, set up the cross entropy loss and step backward
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_mim), y)
        loss.backward()

        # Calculate the perturbation using the current and previous gradient
        grad = X_mim.grad.data / torch.mean(
            torch.abs(X_mim.grad.data), [1, 2, 3], keepdim=True
        )
        previous_grad = decay_factor * previous_grad + grad
        X_mim = Variable(
            X_mim.data + step_size * previous_grad.sign(), requires_grad=True
        )
        eta = torch.clamp(X_mim.data - X.data, -epsilon, epsilon)

        # Add the perturbation and clamp x_mim
        X_mim = Variable(X.data + eta, requires_grad=True)
        X_mim = Variable(torch.clamp(X_mim, 0, 1.0), requires_grad=True)

    # Calculates the MIM error between the prediction and correct
    # classification
    err_mim = (model(X_mim).data.max(1)[1] != y.data).float().sum()

    return err, err_mim


def cw_whitebox(
    model,
    X,
    y,
    device,
    dataset,
    epsilon=0.031,
    num_steps=20,
    step_size=2.0 / 255.0,
):
    """
    Attacks the specified image X using the CW attack and returns the
    adversarial example

    :param model: model being attacked
    :param X: image being attacked
    :param y: correct label of the image being attacked
    :param device: current device
    :param epsilon: epsilon size for the PGD attack
    :param num_steps: number of perturbation steps
    :param step_size: step size of perturbation steps

    :return adversarial example found with the CW attack
    """

    # Create X_pgd basis by duplicating X as a variable
    X_pgd = Variable(X.data, requires_grad=True)

    # If adding random, create random noice between - and + epsilon and add to
    # X_pgd
    random_noise = (
        torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    )
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    # For each perturbation step:
    for _ in range(num_steps):
        # Create the SGD optimizer and zero the gradients
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        # With gradients, set up the CW loss and step backward
        with torch.enable_grad():
            loss = CWLoss(100 if dataset == "cifar100" else 10)(
                model(X_pgd), y
            )
        loss.backward()

        # Calculate the perturbation eta as the step size in the gradient
        # direction of X_pgd
        eta = step_size * X_pgd.grad.data.sign()

        # Add the perturbation to X_pgd
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)

        # Set the perturbation to the difference between X and X_adv,
        # clamped by +/- epsilon
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)

        # Add the new perturbation to X_pgd again
        X_pgd = Variable(X.data + eta, requires_grad=True)

        # Clamp X_gd to be between 0 and 1, as a Variable
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd
