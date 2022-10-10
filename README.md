# Linearised NNs

to install, in the root directory, run the following -

## Cloning the Repository

Since the repository uses submodules, it is recommended to clone the repository with the following command:

```bash
git clone --recursive linearised-nns
git submodule update --init --recursive
```

```bash
sudo apt-get install python3.9-venv
python3.9 -m venv ~/.virtualenvs/sampled
source ~/.virtualenvs/sampled/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python3.9 -m ipykernel install --user --name=sampled
```
