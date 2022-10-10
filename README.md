# Linearised NNs

This repository includes code and experiments for the paper [Sampling-based inference for large linear models, with application to linearised Laplace]().


## Run experiments

As an example, to run stochastic EM for a linearised Laplace model using the LeNetSmall architecture on the MNIST dataset on a Google Cloud TPU VM, run the following command:

```bash
python src/em_trainer.py --config experiments/mnist_gloud_em.py
```
## Cloning the Repository

Since the repository uses submodules, it is recommended to clone the repository with the following command:

```bash
git clone --recursive sampled-laplace
git submodule update --init --recursive
```

## Installation Instructions

```bash
sudo apt-get install python3.9-venv
python3.9 -m venv ~/.virtualenvs/sampled
source ~/.virtualenvs/sampled/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python3.9 -m ipykernel install --user --name=sampled
```
