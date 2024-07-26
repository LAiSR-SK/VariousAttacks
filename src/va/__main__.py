# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
from argparse import ArgumentParser, Namespace
from typing import Dict, Final

from va import VaTraining


def get_args() -> Namespace:
    parser: Final[ArgumentParser] = ArgumentParser(
        prog="Various Attacks CLI tool",
        description="Run the various attacks method on a model of your choice",
    )
    parser.add_argument(
        "model",
        choices=["res18", "res34", "res50", "res101", "res152", "wideres34"],
        type=str,
        help="The model to use",
    )
    parser.add_argument(
        "dataset",
        choices=["cifar10", "cifar100"],
        type=str,
        help="The dataset to use",
    )

    parser.add_argument(
        "--batch-size", type=int, help="Size of training and testing batches"
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--weight-decay", type=float, help="Weight decay constant"
    )
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument(
        "--momentum", type=float, help="Amount of momentum for optimizer"
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed to use for training"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        help="Number of batches to log to the terminal per",
    )
    parser.add_argument(
        "--model-dir", type=str, help="Directory to save models to"
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        help="Number of epochs to save a model checkpoint per",
    )
    parser.add_argument(
        "--swa-warmup",
        type=int,
        help="Number of epochs to apply stochastic weight averaging over",
    )
    parser.add_argument(
        "--swa",
        type=bool,
        help="True to use stochastic weight averaging, False otherwise",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=["decay", "scheduled", "cosine", "cyclic-5", "cyclic-10"],
        type=str,
        help="Learning rate scheduler for training",
    )

    return parser.parse_args()


def main() -> None:
    args: Final[Namespace] = get_args()
    _args: Dict = vars(args)
    _args = {k: v for k, v in _args.items() if v is not None}

    VaTraining(**_args)()


if __name__ == "__main__":
    main()
