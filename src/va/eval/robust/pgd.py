# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
import torch
from torch import nn, optim
from torch.autograd import Variable


def pgd_whitebox_eval(
    model, X, y, device, epsilon=0.031, num_steps=20, step_size=2.0 / 255.0
):
    """
    Evaluates the model by perturbing an image using the PGD attack.

    :param model: model being attacked
    :param X: image being attacked
    :param y: correct label of the image being attacked
    :param device: current device
    :param epsilon: epsilon size for the PGD attack
    :param num_steps: number of perturbation steps
    :param step_size: step size of perturbation steps

    :return clean error and PGD error
    """

    # Calculates clean error of image classification
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()

    # Create X_pgd basis by duplicating X as a variable
    X_pgd = Variable(X.data, requires_grad=True)

    # If specified, create random noice between - and + epsilon and add to
    # X_pgd
    random_noise = (
        torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    )
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    # For each perturbation step
    for _ in range(num_steps):
        # Create the SGD optimizer and zero the gradients
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        # With gradients, set up the cross entropy loss and step backward
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()

        # Calculate the perturbation eta as the step size in the gradient
        # direction of X_pgd
        eta = step_size * X_pgd.grad.data.sign()

        # Add the perturbation to X_pgd
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)

        # Set the perturbation to the difference between X and X_adv, clamped
        # by +/- epsilon
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)

        # Add the new perturbation to X_pgd again
        X_pgd = Variable(X.data + eta, requires_grad=True)

        # Clamp X_gd to be between 0 and 1, as a Variable
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    # Calculate the error between the PGD-perturbed prediction and correct
    # classification
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()

    return err, err_pgd
