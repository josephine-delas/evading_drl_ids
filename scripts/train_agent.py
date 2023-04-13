##########################################################
# Script for training an IDS agent from the command line #
##########################################################
import os
import argparse

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

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
    args = parser.parse_args()

    dataset= args.data
    model = args.model
    hidden_layers = args.layers
    nb_units = args.units if args.units>0 else "custom"
    epochs = args.epochs # each epochs contains training_set.size steps
    
    # Workspace config parameters
    device_name = 'cpu' 
    seed = 0

    # Logging parameters
    log_dir = "../logs"
    output_dir = os.path.join(log_dir, 'train', dataset, model, str(hidden_layers)+'_layer', str(nb_units))

    ####----Device----######
    
    device = config_device(device_name)

    #### Seed #####

    config_seed(seed)

    ####----Environment----####

    nb_proc = 4 # Number of vectorized environments, to accelerate training
    training_env = make_training_env(dataset)() # make_training_env returns a callable function, the last '()' is important
    vectorized_training_env = make_multi_proc_training_env(nb_proc=nb_proc, dataset=dataset)
    testing_env = make_testing_env(dataset)() 
    obs_shape = testing_env.reset().shape

    ####----Agent----####

    agent = Agent(vectorized_training_env, obs_shape, hidden_layers=hidden_layers, nb_units = nb_units,
                   model=model, device=device, seed=seed)

    ####----Training----####

    agent.learn(testing_env, n_envs=nb_proc, save_dir=output_dir, num_epoch=epochs)

    ####----Saving----####
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    agent.save(output_dir + '/last_model.zip')

    print('Done.')