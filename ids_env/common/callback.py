######################################################
# SB3 custom callbacks for monitoring agent training #
######################################################

import numpy as np
from sklearn.metrics import f1_score

from stable_baselines3.common.callbacks import EventCallback
import wandb

from ids_env.common.utils import calcul_rates, print_stats


class CustomWandbCallback(EventCallback):
    '''
    Custom callback to log training data in wandb and save best model.
    '''
    def __init__(self, training_env, eval_env,  eval_freq, save_dir=None, verbose=0):
        '''
        Params:
        -------
            -training_env: gym.Env
                Agent's training environment.
            -eval_env: gym.Env
                Environment used for evaluation (testing set).
            -eval_freq: int
                Callback frequency.
            -save_dir: str
                Directory used to save best model.
        '''
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.states_eval = np.array(self.eval_env.X, dtype='float32')
        self.dict_attack=dict((self.eval_env.attack_types[i], i) for i in range(len(self.eval_env.attack_types)))
        self.labels_test=self.eval_env.y.replace(self.dict_attack) # int values
        self.labels_test_bin = np.sign(self.labels_test) # For binary evaluation

        self.states_train = np.array(training_env.get_attr('X')[0], dtype='float32')
        self.labels_train  = training_env.get_attr('y')[0].replace(self.dict_attack)
        self.labels_train_bin = np.sign(self.labels_train)

        self.eval_freq=eval_freq
        self.save_dir=save_dir
    
    def _on_step(self) -> bool:

        if (self.eval_freq > 0 and self.n_calls % self.eval_freq == 0):

            ## Training metrics ##
            actions_train = self.model.predict(self.states_train)[0]
            actions_train_bin = np.sign(actions_train)
            fpr_train, fnr_train = calcul_rates(self.labels_train, actions_train)
            f1_avg_train = f1_score(self.labels_train, actions_train, average='weighted') # average f1 score over all classes
            f1_bin_train = f1_score(self.labels_train_bin, actions_train_bin) # binary f1 score (normal/attack)

            ## Eval metrics ##
            actions = self.model.predict(self.states_eval)[0]
            actions_bin = np.sign(actions)
            fpr, fnr = calcul_rates(self.labels_test, actions)
            f1_avg = f1_score(self.labels_test, actions, average='weighted')
            f1_bin = f1_score(self.labels_test_bin, actions_bin)

            wandb.log({
                'train':{
                    "FPR":fpr_train,
                    "FNR":fnr_train,
                    "F1 score (avg)":f1_avg_train,
                    "F1 score (bin)":f1_bin_train
                    },
                'eval':{
                    "FPR":fpr,
                    "FNR":fnr,
                    "F1 score (avg)":f1_avg,
                    "F1 score (bin)":f1_bin
                    },
                'epoch':self.n_calls // self.eval_freq})
        
            ## Saving best model ##
            if f1_bin_train > wandb.run.summary["best_f1_bin_train"]:
                wandb.run.summary["best_f1_bin_train"] = f1_bin_train
                self.model.save(self.save_dir + '/best_model_f1_bin.zip')

    
            if self.verbose == 1:
                _,_,_ = print_stats(self.eval_env.attack_types, actions, self.labels_test)

        return True

