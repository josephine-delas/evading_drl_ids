#########################################################
# Building functions for custom environment in sb3      #
#########################################################
from typing import Callable

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from ids_env.common.ids_env import CustomIDSEnv

awid_train_path = "../datasets/AWID/formated_AWID_train.parquet"
awid_test_path = "../datasets/AWID/formated_AWID_test.parquet"

kdd_train_path = "../datasets/KDD/formated_KDD_train.parquet"
kdd_test_path = "../datasets/KDD/formated_KDD_test.parquet"




def make_training_env(dataset: str = "KDD", binary=False) -> Callable:
    '''
    Calls the CustomIDSEnv class to make a training environment
     - dataset is either "AWID" or "KDD", depending on the wanted dataset
     - binary:bin
        If True, binary classification, else multi-class
    '''
    def _init():
        if dataset == "AWID" : 
            train_path = awid_train_path
        elif dataset == "KDD" : 
            train_path = kdd_train_path
        else : 
            raise ValueError("Unknown Dataset")

        return CustomIDSEnv('train', train_path,data=dataset, binary=binary)
    return _init


def make_testing_env(dataset: str = "KDD", binary=False) -> Callable:
    '''
    Calls the CustomIDSEnv class to make a testing environment
    -dataset: str
        Either "AWID" or "KDD", depending on the wanted dataset
    - binary: bin
        If True, binary classification, else multi-class
    '''
    def _init():
        if dataset == "AWID" : 
            test_path = awid_test_path
        elif dataset == "KDD" : 
            test_path = kdd_test_path
        else : 
            raise ValueError("Unknown Dataset")

        return Monitor(CustomIDSEnv('test', test_path, data=dataset, binary=binary))
    return _init

def make_multi_proc_training_env(nb_proc : int, dataset='KDD', binary=False):
    '''
    Vectorizing environement for multi-process training with stable-baselines3
    -nb_proc : int 
        number of wanted subprocesses
    - binary: bin
        If True, binary classification, else multi-class
    '''
    return SubprocVecEnv([make_training_env(dataset=dataset, binary=binary) for _ in range(nb_proc)])
