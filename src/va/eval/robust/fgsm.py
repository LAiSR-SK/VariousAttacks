# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
import torch
from torch import nn, optim
from torch.autograd import Variable


def fgsm_whitebox_eval(model, X, y, epsilon=0.031):
    """
    Evaluates the model by perturbing an image using the FGSM attack.

    :param model: model being attacked
    :param X: image being attacked
    :param y: correct label of the image being attacked
    :param epsilon: epsilon size for the FGSM attack

    :return clean error and FGSM error
    """

    # Calculates clean error of image classification
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()

    # Create X_fgsm basis by duplicating X as a variable
    X_fgsm = Variable(X.data, requires_grad=True)

    # Create the SGD optimizer and zero the gradients
    opt = optim.SGD([X_fgsm], lr=1e-3)
    opt.zero_grad()

    # With gradients, set up the cross entropy loss and step backward
    with torch.enable_grad():
        loss = nn.CrossEntropyLoss()(model(X_fgsm), y)
    loss.backward()

    # Discover X_fgsm by adding epsilon in the gradient direction and clamping
    # the sample
    X_fgsm = Variable(
        torch.clamp(X_fgsm.data + epsilon * X_fgsm.grad.data.sign(), 0.0, 1.0),
        requires_grad=True,
    )

    # Calculates the FGSM error between the prediction and correct
    # classification
    err_fgsm = (model(X_fgsm).data.max(1)[1] != y.data).float().sum()

    return err, err_fgsm
