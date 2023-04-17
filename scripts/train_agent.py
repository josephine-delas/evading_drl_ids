##########################################################
# Script for training an IDS agent from the command line #
##########################################################
import os
import argparse

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

import wandb

from ids_env.common.config_workspace import config_device, config_seed
from ids_env.common.config_ids_env import make_training_env, make_multi_proc_training_env, make_testing_env
from ids_env.common.config_agent import Agent

if __name__=='__main__':

    ####----Parameters----####
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True, default="KDD", help="Dataset to use: 'KDD' or 'AWID'.")
    parser.add_argument("-m", "--model", required=True, default="DQN", help="RL algorithm: 'DQN' or 'PPO'.")
    parser.add_argument("-l", "--layers", required=True, default=1, type=int, help="Number of hidden layers in the policy.")
    parser.add_argument("-u", "--units", required=True, default=64, type=int, help="Number of units in each layer of the policy. If -1, then the network is composed with layers of increasing size.")
    parser.add_argument("-e", "--epochs", required=True, default=10, type=int, help="Number of epochs to train the agent.")
    parser.add_argument("-w", "--wandb", default=1, type=int, help="Activate W&B tracking during training (1) or not (0)")
    parser.add_argument("-p", "--nb_proc", default=4, type=int, help="Number of vectorized environments for training")
    parser.add_argument("-b", "--binary", default=1, type=int, help="Binary classification (True) or multi-class (False)")
    args = parser.parse_args()

    dataset= args.data
    model = args.model
    hidden_layers = args.layers
    nb_units = args.units if args.units>0 else "custom"
    epochs = args.epochs # each epochs contains training_set.size steps
    wandb_on = args.wandb
    nb_proc = args.nb_proc # Number of vectorized environments, to accelerate training
    binary = args.binary
    
    # Workspace config parameters
    device_name = 'cpu' 
    seed = 0

    # Logging parameters
    log_dir = "../logs"
    output_dir = os.path.join(log_dir, 'train', dataset, model, str(hidden_layers)+'_layer', str(nb_units))

    ####---W&B----####
    if wandb_on:
        wandb.init(
            project="evading_drl_ids", # do not change
            tags = [dataset, model],
            name=dataset + '_' + model + '_' + str(hidden_layers) + '_' + str(nb_units), # name of the run
            job_type='train', 
            config={"dataset": dataset, # more information about the run (useful for grouping/filtering)
                    "model": model,
                    "nb_hidden_layers":hidden_layers,
                    "nb_units":nb_units,
                    "epochs": epochs}
        )

        wandb.run.summary["best_f1_bin_train"] = 0  

    ####----Device----######
    
    device = config_device(device_name)

    #### Seed #####

    config_seed(seed)

    ####----Environment----####
    
    training_env = make_training_env(dataset, binary=binary)() # make_training_env returns a callable function, the last '()' is important
    vectorized_training_env = make_multi_proc_training_env(nb_proc=nb_proc, dataset=dataset, binary=binary)
    testing_env = make_testing_env(dataset, binary=binary)() 
    obs_shape = testing_env.reset().shape

    ####----Agent----####

    agent = Agent(vectorized_training_env, obs_shape, hidden_layers=hidden_layers, nb_units = nb_units,
                   model=model, device=device, seed=seed, wandb_on=wandb_on)

    if wandb_on:
        wandb.run.summary["network architecture"] = agent.model.policy.net_arch

    ####----Training----####

    agent.learn(testing_env, n_envs=nb_proc, save_dir=output_dir, num_epoch=epochs)

    ####----Saving----####
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if binary :
        agent.save(output_dir + '/last_model_binary.zip')
    else:
        agent.save(output_dir + '/last_model.zip')

    print('Done.')