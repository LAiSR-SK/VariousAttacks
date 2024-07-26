# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
from torch.autograd import Variable


def clean(model, X, y):
    """
    Evaluates a model on a clean sample.

    :param model: classifier to be evaluated
    :param X: image
    :param y: correct classification of image

    :return the error between the model prediction of the image and the
    correct classification
    """

    out = model(X)  # model prediction
    err = (
        (out.data.max(1)[1] != y.data).float().sum()
    )  # error between prediction and classification
    return err


def eval_clean(model, device, data_loader, name, ds_name, f):
    """
    Calculates the natural error of the model on either the training
    or test set of data

    :param model: trained model to be evaluated
    :param device: the device the model is set on
    :param data_loader: data loader containing the dataset to test
    :param name: 'train' or 'test', denoting which dataset is being tested
    :param ds_name: name of dataset, cifar10 or cifar100
    :param f: file to print results to
    """

    model.eval()

    # Set the total errors to 0 for all attacks but AA
    total_err = 0

    # Run tests for each element in the test set
    for sample in data_loader:
        # Set up the data/X and target/y correctly for evaluation
        if ds_name == "cifar100":
            data, target, _ = sample
        else:
            data, target = sample
        data, target = data.to(device), target.to(device)
        X, y = Variable(data, requires_grad=True), Variable(target)

        # Calculate the natural error for each attack
        err = clean(model, X, y)

        # Add the losses to the total loss for each attack
        total_err += err

    # Write the total losses to the file
    if name == "test":
        # Convert the clean loss to clean accuracy %
        clean_total = int(total_err)
        clean_acc = (10000 - clean_total) / 100
        print("Clean Accuracy (Test): " + str(clean_acc) + "%")
        f.write("Clean Accuracy (Test): " + str(clean_acc) + "%\n")
    elif name == "train":
        # Convert the clean loss to clean accuracy %
        clean_total = int(total_err)
        clean_acc = (50000 - clean_total) / 500
        print("Clean Accuracy (Train): " + str(clean_acc) + "%")
        f.write("Clean Accuracy (Train): " + str(clean_acc) + "%\n")
    else:
        raise NotImplementedError
