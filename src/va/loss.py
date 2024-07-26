# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
import torch
from torch import nn
from torch.autograd import Variable

from va.attack import create_attack


def CWLoss(output, target, confidence=0):
    """
    CW loss (Marging loss).
    """
    num_classes = output.shape[-1]
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1.0 - target_var) * output - target_var * 10000.0).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.0)
    loss = torch.sum(loss)
    return loss


def standardat_loss(model, x, y, optimizer, keep_clean=False):
    """
    Basic loss function from Madry et al. calculating an adversarial sample
    and the loss resulting.

    :param model: trained image classifier
    :param x: batch of clean of images
    :param y: labels for images in x
    :param optimizer: optimizer associated with the model
    :param attack: attack used to perturb clean images
    :param keep_clean: True if clean examples are kept

    :return cross-entropy loss between the adversarial output and correct class
    """

    # Calculate the adversarial example based on the atts
    criterion = nn.CrossEntropyLoss()
    attack = create_attack(model, criterion, "linf-pgd", 0.1, 40, 0.01)
    x_adv, _ = attack.perturb(x, y)

    optimizer.zero_grad()  # zero out the gradients

    # Correctly format x_adv and y_adv
    if keep_clean:
        x_adv = torch.cat((x, x_adv), dim=0)
        y_adv = torch.cat((y, y), dim=0)
    else:
        y_adv = y

    # Use adversarial output to calculate the cross-entropy loss with correct
    # labels
    out = model(x_adv)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(out, y_adv)

    return loss


def standard_loss(model, x, y, optimizer):
    """
    Standard training loss function with no adversarial attack.

    :param model: image classifier
    :param x: batch of clean images
    :param y: correct labels for the clean images in x
    :param optimizer: optimizer for the model

    :return loss between correct labels and model output for the batch x
    """
    optimizer.zero_grad()  # zero out the gradients

    # Calculate cross-entropy loss between model output and correct labels
    out = model(x)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(out, y)

    # Format predictions and batch metrics for return
    preds = out.detach()
    batch_metrics = {"loss": loss.item(), "clean_acc": accuracy(y, preds)}

    return loss, batch_metrics


def accuracy(true, preds):
    """
    Computes multi-class accuracy.

    :param true: true labels
    :param preds: predicted labels
    :return multi-class accuracy
    """
    accuracy = (
        torch.softmax(preds, dim=1).argmax(dim=1) == true
    ).sum().float() / float(true.size(0))
    return accuracy.item()
