#################################################################
# Script for training multiple IDS agents from the command line #
#################################################################
import os
import argparse
import numpy as np
from sklearn.metrics import f1_score
import torch.nn as nn
import wandb
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))


from ids_env.common.config_workspace import config_device, config_seed
from ids_env.common.config_ids_env import make_training_env, make_multi_proc_training_env, make_testing_env
from ids_env.common.config_agent import Agent
from ids_env.common.utils import calcul_rates

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
    epsilon_range=[0., 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

    # Workspace config parameters
    device_name = 'cpu' 
    seed = 0

    # Logging parameters
    log_dir = "../logs"
    output_dir = os.path.join(log_dir, 'train', dataset, model, str(hidden_layers)+'_layer', str(nb_units))

    ####----Device----######
    
    device = config_device(device_name)
    config_seed(seed)

    ####----Evaluation Variables----####

    testing_env = make_testing_env(dataset, binary=binary)() 
    training_env= make_training_env(dataset, binary=binary)()
    obs_shape = testing_env.reset().shape

    test_set=np.array(testing_env.X, dtype='float32')
    train_set=np.array(training_env.X, dtype='float32')
    dict_attack=dict((testing_env.attack_types[i], i) for i in range(len(testing_env.attack_types)))
    test_labels=testing_env.y.replace(dict_attack)
    train_labels=training_env.y.replace(dict_attack)
    nb_class = len(testing_env.attack_types)
    if binary:
        test_labels = np.sign(test_labels)
        train_labels = np.sign(train_labels)
        nb_class=2


    ####---Loop----####

    for i in range(nb_agents):
        ####----W&B----####
        wandb.init(
            project="evading_drl_ids", # do not change
            tags = [dataset, model],
            name=dataset + '_' + model + '_' + str(hidden_layers) + '_' + str(nb_units), # name of the run
            job_type='train', 
            config={"dataset": dataset, # more information about the run (useful for grouping/filtering)
                    "model": model,
                    "nb_hidden_layers":hidden_layers,
                    "nb_units":nb_units,
                    "epochs": epochs,
                    "agent":i,
                    "binary":binary}
        )

        wandb.define_metric(
            name='training_metrics',
            step_metric='epoch'
            )
        
        print('Started training of agent {} / {}'.format(i+1, nb_agents))

        vectorized_training_env = make_multi_proc_training_env(nb_proc=nb_proc, dataset=dataset, binary=binary)

       # Creation of the agent

        args_callback = {
            'test_set':test_set,
            'train_set':train_set,
            'dict_attack':dict_attack,
            'test_labels':test_labels,
            'train_labels':train_labels,
            'binary':binary
        }

        agent = Agent(vectorized_training_env, obs_shape, args_callback=args_callback, hidden_layers=hidden_layers, nb_units = nb_units,
                   model=model, device=device, seed=seed)

        # Training
        print('Training...')
        agent.learn(testing_env, n_envs=nb_proc, save_dir=output_dir, num_epoch=epochs)
        print('Training done')

        ####----Adversarial Attack----####

        pytorch_model = agent.get_pytorch_model()
        classifier = PyTorchClassifier(model = pytorch_model, loss=nn.HuberLoss(), 
                                       input_shape=test_set.shape[1], nb_classes=nb_class)

        for epsilon in epsilon_range:

            print("FGSM Attack...")
            fgm = FastGradientMethod(classifier,
                                         norm=np.inf,
                                         eps=epsilon,
                                         targeted=True,
                                         num_random_init=0,
                                         batch_size=128,
                                         )
            adversarial_examples = fgm.generate(x=test_set, y=np.hstack((np.ones((test_set.shape[0], 1)), np.zeros((test_set.shape[0], nb_class-1)))).astype('float32'))# we assume the normal class is the first column
            adversarial_actions = agent.model.predict(adversarial_examples, deterministic=True)[0]
            fpr, fnr = calcul_rates(test_labels, adversarial_actions)
            if binary:
                f1 = f1_score(test_labels, adversarial_actions)
            else:
                f1 = f1_score(test_labels, adversarial_actions, average='weighted')
            
            wandb.log({'FGSM_metric':{
                    "FPR":fpr,
                    "FNR":fnr,
                    "F1 score":f1
                    },        
                    "epsilon":epsilon            
                    })
            
            print("BIM Attack...")
            bim = BasicIterativeMethod(classifier, 
                                           eps=epsilon, 
                                           eps_step=epsilon/100.,
                                           max_iter=100, 
                                           targeted=True, 
                                           batch_size=128)
            
            adversarial_examples = bim.generate(x=test_set, y=np.hstack((np.ones((test_set.shape[0], 1)), np.zeros((test_set.shape[0], nb_class-1)))).astype('float32'))# we assume the normal class is the first column
            adversarial_actions = agent.model.predict(adversarial_examples, deterministic=True)[0]
            fpr, fnr = calcul_rates(test_labels, adversarial_actions)
            if binary:
                f1 = f1_score(test_labels, adversarial_actions)
            else:
                f1 = f1_score(test_labels, adversarial_actions, average='weighted')

            wandb.log({'BIM_metric':{
                            "FPR":fpr,
                            "FNR":fnr,
                            "F1 score":f1
                            },        
                        "epsilon":epsilon            
                        })  
            # Verbose 
            #print('Adversarial Attack:')
            #if binary : 
            #    print_stats(['Normal', 'Attack'], test_labels, adversarial_actions)
            #else:
            #    print_stats(testing_env.attack_types, test_labels, adversarial_actions)
        

        print('Attack done.')
        wandb.finish()
    # Saving model and evaluation metrics
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    agent.save(output_dir + '/last_model.zip')
    #test_fpr.tofile(output_dir+'/test_fpr.np')
    #test_fnr.tofile(output_dir+'/test_fnr.np')
    #test_f1.tofile(output_dir+'/test_f1_avg.np')
    #adv_fpr.tofile(output_dir+'/adv_fpr.np')
    #adv_fnr.tofile(output_dir+'/adv_fnr.np')
    #adv_f1.tofile(output_dir+'/adv_f1_avg.np')

