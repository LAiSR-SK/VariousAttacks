# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
import torch
from torch import nn, optim
from torch.autograd import Variable


def mim_whitebox(
    model,
    X,
    y,
    device,
    epsilon=0.031,
    num_steps=20,
    step_size=2.0 / 255.0,
    decay_factor=1.0,
):
    """
    Attacks the specified image X using the MIM attack and returns the
    adversarial example

    :param model: model being attacked
    :param X: image being attacked
    :param y: correct label of the image being attacked
    :param device: current device
    :param epsilon: epsilon size for the PGD attack
    :param num_steps: number of perturbation steps
    :param step_size: step size of perturbation steps
    :param decay_factor: factor of decay for gradients

    :return adversarial example found with the MIM attack
    """

    # Create X_pgd basis by duplicating X as a variable
    X_pgd = Variable(X.data, requires_grad=True)

    # If adding random, create random noice between - and + epsilon and add to
    # X_pgd
    random_noise = (
        torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    )
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    # Set the previous gradient to a tensor of 0s in the shape of X
    previous_grad = torch.zeros_like(X.data)

    # For each perturbation step:
    for _ in range(num_steps):
        # Create the SGD optimizer and zero the gradients
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        # With gradients, set up the cross-entropy loss and step backward
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()

        # Calculate the gradient by dividing it by the average
        grad = X_pgd.grad.data / torch.mean(
            torch.abs(X_pgd.grad.data), [1, 2, 3], keepdim=True
        )

        # Calculate the previous gradient by multiplying it by the decay
        # factor and adding to the grad
        previous_grad = decay_factor * previous_grad + grad

        # Perturb X_pgd in the direction of the previous grad, by the step size
        X_pgd = Variable(
            X_pgd.data + step_size * previous_grad.sign(), requires_grad=True
        )

        # Set the perturbation to the difference between X and X_adv, clamped
        # by +/- epsilon
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)

        # Add the new perturbation to X_pgd againt
        X_pgd = Variable(X.data + eta, requires_grad=True)

        # Clamp X_gd to be between 0 and 1, as a Variable
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd
