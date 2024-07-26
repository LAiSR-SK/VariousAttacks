![](asset/repo/image/banner.webp)
# Various Attacks
![License](https://img.shields.io/github/license/LAiSR-SK/VariousAttacks) ![Code Style](https://img.shields.io/badge/code_style-Ruff-orange)

Various Attacks is a novel adversarial training method based on Adversarial Distribution Training. <!--TODO: Someone needs to add to this -->

## 📊 Our Results
Below are the robustness of adversarial training defense methods on the CIFAR-10 and CIFAR-100 datasets. The bold results are the highest among the upper portion of the tables. The lower tables contain defense methods that, while unusually high in some categories, suffer from serious vulnerabilities in others.

<!--TODO: Is there a reason the original tables were formatted with a * for "better than the majority of other methods"? This seems like a strange way to present results -->
### CIFAR-10
| Defense Method| Clean     | FGSM      | MIM       | CW        | PGD-20    | PGD-40    | AA        |
|---------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| Clean         | **95.09%**| 26.08%    | 00.00%    | 00.00%    | 00.00%    | 00.00%    | 00.00%    |
| Standard AT   | 86.89%    | 57.87%    | 51.09%    | 49.90%    | 51.68%    | 51.56%    | 48.32%    |
| TRADES        | 84.58%    | 60.18%    | 54.97%    | 52.95%    | 55.53%    | 55.40%    | 52.02%    |
| ADT           | 83.63%    | 56.90%    | 49.93%    | 48.73%    | 50.51%    | 50.29%    | 45.98%    |
| GAIRAT        | 85.74%    | 56.69%    | 56.81%    | 44.48%    | 58.63%    | 58.67%    | 42.48%    |
| LAS-AT        | 87.34%    | 62.11%    | 55.81%    | 54.72%    | 56.39%    | 56.23%    | 53.03%    |
| DNR (C)       | 87.48%    | 55.74%    | 46.65%    | 44.76%    | 47.41%    | 47.00%    | 42.40%    |
| DNR (I)       | 87.31%    | 54.69%    | 45.80%    | 43.07%    | 46.42%    | 46.18%    | 40.97%    |
| YOPO          | 86.34%    | 55.26%    | 48.17%    | 47.71%    | 48.72%    | 48.37%    | 44.93%    |
| FAT           | 89.06%    | 58.81%    | 48.78%    | 47.29%    | 48.28%    | 47.96%    | 44.42%    |
| **VA**        | 91.15%    | **64.98%**| **61.69%**| **56.52%**| **68.71%**|**68.58%** | **64.74%**|

| Defense Method| Clean     | FGSM      | MIM       | CW        | PGD-20    | PGD-40    | AA        |
|---------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| Curriculum AT | 89.92%    | 78.55%    | 03.83%    | 35.40%    | 40.27%    | 26.01%    | 00.14%    |
| Customized AT | 94.09%    | 81.29%    | 74.13%    | 58.79%    | 68.47%    | 66.40%    | 21.68%    |

### CIFAR-100
| Defense Method| Clean     | FGSM      | MIM       | CW        | PGD-20    | PGD-40    | AA        |
|---------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| Clean         | **76.65%**| 08.77%    | 00.01%    | 00.00%    | 00.00%%   | 00.00%    | 00.00%    |
| Standard AT   | 60.73%    | 31.08%    | 27.47%    | 26.13%    | 27.79%    | 27.64%    | 24.90%    |
| TRADES        | 52.06%    | 27.88%    | 25.79%    | 22.75%    | 26.52%    | 26.53%    | 21.93%    |
| ADT           | 57.72%    | 30.50%    | 24.76%    | 23.88%    | 25.47%    | 25.29%    | 21.53%    |
| GAIRAT        | 60.06%    | 28.61%    | 24.66%    | 23.11%    | 25.08%    | 25.01%    | 21.28%    |
| LAS-AT        | 59.22%    | 32.00%    | 26.39%    | 23.21%    | 25.75%    | 25.45%    | 21.96%    |
| YOPO          | 62.31%    | 28.51%    | 24.23%    | 23.57%    | 24.48%    | 24.31%    | 21.37%    |
| FAT           | 65.09%    | 29.18%    | 23.24%    | 23.02%    | 23.25%    | 23.14%    | 21.44%    |
| **VA**        | 61.90%    | **32.77%**| **29.55%**| **28.13%**| **29.92%**| **30.11%**| **25.93%**|

| Defense Method| Clean     | FGSM      | MIM       | CW        | PGD-20    | PGD-40    | AA        |
|---------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| Curriculum AT | 64.73%    | 70.55%    | 00.98%    | 08.92%    | 20.32%    | 12.97%    | 00.03%    |
| Customized AT | 73.14%    | 45.99%    | 37.23%    | 07.23%    | 34.96%    | 33.83%    | 11.59%    |

## 🔬 Replicating our Work
### Cloning the Repository
To clone our repo, simply run:
> git clone https://github.com/LAiSR-SK/VariousAttacks.git <PATH_TO_YOUR_CLONE>

### Creating the Environment
To create the [conda environment](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html), run:
> conda env create -f environment.yml

The environment specified in [`environment.yml`](environment.yml) is for a machine running Windows 11. If you are running Linux, the following core dependencies should be installed:
- `pytorch-cuda` version 11.8
- `torchvision` version 0.16.1
- `autattack` version 0.1 (installable with pip)

From the `base`, `pytorch` and `nvidia` channels.

### Running the Code
Any code you write should work from the `script/` directory. Our code can also be run by moving the contents of the `src/` directory into your PYTHONPATH.

In addition to our code interface, you can use our command line interface to train a Resnet or WideReset model. To invoke the interface, cd to `script/` and run:
> python3 -m va

## 📝 Citing us
See [`CITATION.cff`](CITATION.cff) or the side pane of this repository for details on citing our work.
