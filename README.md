# Evading DRL-based Network Intrusion Detection with Adversarial Attack

*Joséphine Delas, Mohamed Amine Merzouk, Christopher Neal*

## Installation

Code runs on Python 3.9.1.

#### Requirements 

Activate the local environment and install the requirements:

```sh
pip install -r requirements.txt
```
#### W&B

Log in to wandb and paste your API key when prompted:

```sh
wandb login
```

## Tests

You can test the preprocessing functions and try to create the gym environment with the scripts from `./tests/`

```sh
cd tests/
python -m test_preprocessing -d <dataset> -s <save> -v <verbose>
# Example
python -m test_preprocessing -d KDD -s 1 -v 1 
```

## Scripts

The `./scripts/` directory contains scripts for training and evaluating IDS agents. 

#### Training script

This script trains an IDS agent with the wanted hyperparameters, showing wandb metrics and saving best and last models into the `./logs/train/`directory.

```sh
cd scripts/
python -m train_agent -d <dataset> -m <model> -l <num_layers> -u <num_units> -e <num_epoch> -w <wandb_on> -p <nb_proc>
# Example
python -m train_agent -d KDD -m DQN -l 1 -u 68 -e 10 -w 1
```

#### Evaluation script

This script loads a pretrained agent from the log directory, raising an error if the file doesn't exist (wrong dataset or no pretraining). It prints its performance metrics : F1 score, false positive rate (number of attacks classified as normal), false negative rate (number of false alarms).

```sh
cd scripts/
python -m eval_agent -d <dataset> -m <model> -l <num_layers> -u <num_units> 
```