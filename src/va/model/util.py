# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
from va.model.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from va.model.wideresnet import WideResNet


def get_model(mod_name, ds_name, device):
    if mod_name == "res18":
        model = ResNet18(num_classes=100 if ds_name == "cifar100" else 10).to(
            device
        )
    elif mod_name == "res34":
        model = ResNet34(num_classes=100 if ds_name == "cifar100" else 10).to(
            device
        )
    elif mod_name == "res50":
        model = ResNet50(num_classes=100 if ds_name == "cifar100" else 10).to(
            device
        )
    elif mod_name == "res101":
        model = ResNet101(num_classes=100 if ds_name == "cifar100" else 10).to(
            device
        )
    elif mod_name == "res152":
        model = ResNet152(num_classes=100 if ds_name == "cifar100" else 10).to(
            device
        )
    elif mod_name == "wideres34":
        model = WideResNet(
            depth=34, num_classes=100 if ds_name == "cifar100" else 10
        ).to(device)
    else:
        raise NotImplementedError

    return model
