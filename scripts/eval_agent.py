##########################################################
# Script for running an IDS agent from the command line #
##########################################################

import os
import argparse
import numpy as np
from sklearn.metrics import f1_score

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))


from ids_env.common.config_workspace import config_device, config_seed
from ids_env.common.config_ids_env import make_testing_env
from ids_env.common.config_agent import Agent
from ids_env.common.utils import calcul_rates, print_stats

if __name__=='__main__':

    ####----Parameters----####

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True, default="KDD", help="Dataset to use: 'KDD' or 'AWID'.")
    parser.add_argument("-m", "--model", required=True, default="DQN", help="RL algorithm: 'DQN' or 'PPO'.")
    parser.add_argument("-l", "--layers", required=True, default=1, type=int, help="Number of hidden layers in the policy.")
    parser.add_argument("-u", "--units", required=True, default=64, type=int, help="Number of units in each layer of the policy. If -1, then the network is composed with layers of increasing size.")

    args = parser.parse_args()

    dataset= args.data
    model = args.model
    hidden_layers = args.layers
    nb_units = args.units if args.units>0 else "custom"

     # Workspace config parameters
    device_name = 'cpu' 
    seed = 0

    # Load parameters
    log_dir = "../logs"
    load_dir = os.path.join(log_dir, 'train', dataset, model, str(hidden_layers)+'_layer', str(nb_units))

    if not os.path.exists(load_dir):
        raise(ValueError("The wanted model doesn't exist yet: try with other hyperparameter values"))

    ####----Device----######
    
    device = config_device(device_name)

    ####----Seed----#####

    config_seed(seed)

    ####----Environment----####

    testing_env = make_testing_env(dataset)() 
    obs_shape = testing_env.reset().shape

    ####----Agent----####

    agent = Agent(testing_env, obs_shape, hidden_layers=hidden_layers, nb_units=nb_units, model=model, device=device, seed=seed)

    agent_name='best_model_f1_bin.zip'
    agent.load(os.path.join(load_dir, agent_name))

    ####----Test----####

    test_set=np.array(testing_env.X, dtype='float32')
    dict_attack=dict((testing_env.attack_types[i], i) for i in range(len(testing_env.attack_types)))
    test_labels=testing_env.y.replace(dict_attack)

    actions = agent.model.predict(test_set)[0]

    fpr, fnr = calcul_rates(test_labels, actions)
    f1_avg = f1_score(test_labels, actions, average='weighted')
    print('\nPerformance of '+ model + ' with ' + str(hidden_layers) + ' layers and ' + str(nb_units) + ' units in each layer on the ' + dataset + ' dataset:')
    print('\nF1 score (avg): '+str(f1_avg))
    print('False positive rate: '+str(fpr))
    print('False negative rate: '+str(fnr))
    print('\nDetailed information: ')
    print_stats(dict_attack, test_labels, actions)






