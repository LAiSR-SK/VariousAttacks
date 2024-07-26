# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
from torch.autograd import Variable

from va.eval.robust.fgsm import fgsm_whitebox_eval
from va.eval.robust.pgd import pgd_whitebox_eval


def robust_eval(model, device, test_loader, ds_name, f):
    """
    Calculates the clean loss and robust  loss against PGD and FGSM attacks

    :param model: trained model to be evaluated
    :param device: the device the model is set on
    :param test_loader: data loader containing the testing dataset
    :param f: the file to write results to

    :return pgd_acc: the accuracy against PGD attack
    """

    model.eval()

    # Set the total errors to 0 for all atts but AA
    clean_total = 0
    pgd_robust_total = 0
    fgsm_robust_total = 0

    # Run tests for each element in the test set
    for sample in test_loader:
        # Set up the data/X and target/y correctly for evaluation
        if ds_name == "cifar100":
            data, target, _ = sample
        else:
            data, target = sample
        data, target = data.to(device), target.to(device)
        X, y = Variable(data, requires_grad=True), Variable(target)

        # Calculate the natural and robust error for each attack
        pgd_err_natural, pgd_err_robust = pgd_whitebox_eval(
            model, X, y, device
        )
        fgsm_err_natural, fgsm_err_robust = fgsm_whitebox_eval(model, X, y)

        # Add the losses to the total loss for each attack
        clean_total += pgd_err_natural
        pgd_robust_total += pgd_err_robust
        fgsm_robust_total += fgsm_err_robust

    # Convert the clean error to clean accuracy %
    clean_total = int(clean_total)
    clean_acc = (10000 - clean_total) / 100

    # Convert the PGD error to clean accuracy %
    pgd_robust_total = int(pgd_robust_total)
    pgd_acc = (10000 - pgd_robust_total) / 100

    # Convert the FGSM error to clean accuracy %
    fgsm_robust_total = int(fgsm_robust_total)
    fgsm_acc = (10000 - fgsm_robust_total) / 100

    # Print out the loss percents
    print("Clean Accuracy (Test): " + str(clean_acc) + "%")
    print("PGD Robust Accuracy: " + str(pgd_acc) + "%")
    print("FGSM Robust Accuracy: " + str(fgsm_acc) + "%")

    # Write loss percents to file
    f.write("Clean Accuracy (Test): " + str(clean_acc) + "%\n")
    f.write("PGD Robust Accuracy: " + str(pgd_acc) + "%\n")
    f.write("FGSM Robust Accuracy: " + str(fgsm_acc) + "%\n")

    return pgd_acc
