# Neural Green's Operators (NGOs)

Code for the paper

> H.A. Melchers\*, J.H.M. Prins\*, M.R.A. Abdelmalik (2026).
> **Neural Green's operators for parametric partial differential equations.**
> *Computer Methods in Applied Mechanics and Engineering* 455, 118893.
> [doi:10.1016/j.cma.2026.118893](https://doi.org/10.1016/j.cma.2026.118893)
> (preprint: [arXiv:2406.01857](https://arxiv.org/abs/2406.01857))
>
> \* Equal contribution.

Neural Green's Operators (NGOs) are neural operators for parametric partial differential equations (PDEs) whose architecture is based on the Green's function representation of the solution. This repository contains the implementation used in the paper, the baseline models it compares against, and two test problems: steady diffusion and time-dependent diffusion.

## Installation

Create a Python (>=3.9) virtual environment, then run from the root of this repository:

```bash
pip install -r requirements.txt
pip install -e .
```

To run the notebooks, select the environment in which you installed the requirements as the Jupyter kernel.

## Quick start

The [`examples`](examples) directory contains two notebooks that can be run from top to bottom:

- [`steadydiffusion.ipynb`](examples/steadydiffusion.ipynb) trains a model NGO for the steady diffusion problem within a few seconds on a GPU, plots a true solution against the model NGO prediction, and plots the prediction error against the length scale of the test data.
- [`timedependentdiffusion.ipynb`](examples/timedependentdiffusion.ipynb) trains a model NGO for the time-dependent diffusion problem within a few minutes on a GPU, animates the true, projected and predicted solutions over time, and plots the prediction error over time.

Both notebooks use the first GPU (`cuda:0`). To run on a CPU, set `hparams['accelerator'] = 'cpu'` and `hparams['device'] = 'cpu'`.

## Training from the command line

Each test problem in [`src/ngo/testproblems`](src/ngo/testproblems) has a `train.py` script in which all hyperparameters are set in the `hparams` dictionary. The model type is selected with `hparams['modeltype']`; the options are `"NN"`, `"DeepONet"`, `"VarMiON"`, `"data NGO"` and `"model NGO"`. Training logs are written in TensorBoard format to the directory set by `logdir`.

Example SLURM job scripts are included (`train.sh`, `train_snellius.sh`); adapt the partition, resources and environment path to your cluster.

## Repository structure

```
examples/                   example notebooks
src/ngo/
  ml/                       network layers, system nets, losses, training and model loading
  numericalanalysis/        basis functions, quadrature rules and linear algebra
  trainingdata/             Gaussian random field generation for the training data
  testproblems/
    steadydiffusion/        data module, neural operator and training script
    timedependentdiffusion/ data module, neural operator and training script
```

## Citation

If you use this code, please cite the paper:

```bibtex
@article{melchers2026neural,
  title   = {Neural {Green's} operators for parametric partial differential equations},
  author  = {Melchers, H. A. and Prins, J. H. M. and Abdelmalik, M. R. A.},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  volume  = {455},
  pages   = {118893},
  year    = {2026},
  doi     = {10.1016/j.cma.2026.118893}
}
```

## License

This project is licensed under the MIT License; see [LICENSE](LICENSE).
