
from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from ids_env.common.config_ids_env import make_training_env, make_multi_proc_training_env, make_testing_env

if __name__=='__main__':

    train_env = make_training_env('KDD')()
    test_env = make_testing_env('KDD')()
    vectorized_training_env = make_multi_proc_training_env(nb_proc=2, dataset='KDD')
    
    print(vectorized_training_env.reset())
