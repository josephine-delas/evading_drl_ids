#################################################################
# Script for training multiple IDS agents from the command line #
#################################################################
import os
import argparse
import numpy as np
from sklearn.metrics import f1_score

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from ids_env.common.config_workspace import config_device, config_seed
from ids_env.common.config_ids_env import make_training_env, make_multi_proc_training_env, make_testing_env
from ids_env.common.config_agent import Agent
from ids_env.common.utils import calcul_rates, print_stats

if __name__=='__main__':

    ####----Parameters----####
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True, default="KDD", help="Dataset to use: 'KDD' or 'AWID'.")
    parser.add_argument("-m", "--model", required=True, default="DQN", help="RL algorithm: 'DQN' or 'PPO'.")
    parser.add_argument("-l", "--layers", required=True, default=1, type=int, help="Number of hidden layers in the policy.")
    parser.add_argument("-u", "--units", required=True, default=64, type=int, help="Number of units in each layer of the policy. If -1, then the network is composed with layers of increasing size.")
    parser.add_argument("-e", "--epochs", required=True, default=10, type=int, help="Number of epochs to train the agent.")
    parser.add_argument("-p", "--nb_proc", default=4, type=int, help="Number of vectorized environments for training")
    parser.add_argument("-n", "--nb_agents", default=1, type=int, help="Number of agents trained in the loop")
    parser.add_argument("-b", "--binary", default=1, type=int, help="Binary classification (1) or multi-class (0)")
    args = parser.parse_args()

    dataset= args.data
    model = args.model
    hidden_layers = args.layers
    nb_units = args.units if args.units>0 else "custom"
    epochs = args.epochs # each epochs contains training_set.size steps
    nb_proc = args.nb_proc # Number of vectorized environments, to accelerate training
    nb_agents = args.nb_agents
    binary = args.binary
    verbose = False

    # Workspace config parameters
    device_name = 'cpu' 
    seed = 0

    # Logging parameters
    log_dir = "../logs"
    output_dir = os.path.join(log_dir, 'train', dataset, model, str(hidden_layers)+'_layer', str(nb_units))

    ####----Device----######
    
    device = config_device(device_name)
    config_seed(seed)

    ####----Evaluation----####

    testing_env = make_testing_env(dataset, binary=binary)() 
    training_env= make_training_env(dataset, binary=binary)()
    obs_shape = testing_env.reset().shape

    test_set=np.array(testing_env.X, dtype='float32')
    train_set=np.array(training_env.X, dtype='float32')
    dict_attack=dict((testing_env.attack_types[i], i) for i in range(len(testing_env.attack_types)))
    test_labels=testing_env.y.replace(dict_attack)
    train_labels=training_env.y.replace(dict_attack)
    if binary:
        test_labels = np.sign(test_labels)
        train_labels = np.sign(train_labels)


    ####---Loop----#
    best_fpr=1
    best_fnr=1
    best_f1_score=0
    test_fpr=np.zeros(nb_agents)
    test_fnr=np.zeros(nb_agents)
    test_f1=np.zeros(nb_agents)
    for i in range(nb_agents):
        print('Started training of agent {} / {}'.format(i, nb_agents))

        vectorized_training_env = make_multi_proc_training_env(nb_proc=nb_proc, dataset=dataset, binary=binary)

       # Creation of the agent

        agent = Agent(vectorized_training_env, obs_shape, hidden_layers=hidden_layers, nb_units = nb_units,
                   model=model, device=device, seed=seed)

        # Training

        agent.learn(testing_env, n_envs=nb_proc, save_dir=output_dir, num_epoch=epochs)

        if model=='DQN':
            agent.model.policy.set_training_mode(False) # Switch to testing mode

        
        # Evaluation on the testing set 

        test_actions = agent.model.predict(test_set, deterministic=True)[0]
        test_fpr[i], test_fnr[i] = calcul_rates(test_labels, test_actions)
        if binary:
            test_f1[i] = f1_score(test_labels, test_actions)
        else:
            test_f1[i] = f1_score(test_labels, test_actions, average='weighted')

        print('Completed training of agent {} / {}'.format(i, nb_agents))

    # Saving model and evaluation metrics
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    agent.save(output_dir + '/last_model.zip')
    test_fpr.tofile(output_dir+'/test_fpr.np')
    test_fnr.tofile(output_dir+'/test_fnr.np')
    test_f1.tofile(output_dir+'/test_f1_avg.np')

