# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
import torch


class Attack:
    """
    Abstract base class for all attack classes.
    Arguments:
        predict (nn.Module): forward pass function.
        loss_fn (nn.Module): loss function.
        clip_min (float): mininum value per input dimension.
        clip_max (float): maximum value per input dimension.
    """

    def __init__(self, predict, loss_fn, clip_min, clip_max):
        self.predict = predict
        self.loss_fn = loss_fn
        self.clip_min = clip_min
        self.clip_max = clip_max

    def perturb(self, x, **kwargs):
        """
        Virtual method for generating the adversarial examples.
        Arguments:
            x (torch.Tensor): the model's input tensor.
            **kwargs: optional parameters used by child classes.
        Returns:
            adversarial examples.
        """
        error = "Sub-classes must implement perturb."
        raise NotImplementedError(error)

    def __call__(self, *args, **kwargs):
        return self.perturb(*args, **kwargs)


class LabelMixin:
    def _get_predicted_label(self, x):
        """
        Compute predicted labels given x. Used to prevent label leaking during
        adversarial training.
        Arguments:
            x (torch.Tensor): the model's input tensor.
        Returns:
            torch.Tensor containing predicted labels.
        """
        with torch.no_grad():
            outputs = self.predict(x)
        _, y = torch.max(outputs, dim=1)
        return y

    def _verify_and_process_inputs(self, x, y):
        if self.targeted:
            assert y is not None

        if not self.targeted and y is None:
            y = self._get_predicted_label(x)

        x = x.detach().clone()
        y = y.detach().clone()
        return x, y
