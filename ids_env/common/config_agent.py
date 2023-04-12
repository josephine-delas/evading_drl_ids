#########################################################
# Agent class based on nn and sb3 modules               #
#########################################################
import numpy as np
import torch
import torch.nn as nn

from stable_baselines3 import PPO, DQN

from ids_env.common.callback import CustomWandbCallback

class Agent(nn.Module):
    '''
    SB3 Agent for the IDS environment, with helper functions to help implement adversarial examples
    '''

    def __init__(self, env, obs_shape, hidden_layers=1, nb_units = 64, model: str = 'DQN', device: str = 'cpu', seed: int = 0, wandb_on=True):
        '''
        Params:
        -------
        -env: gym.Env
            Gym environment in which the agent will evolve.
        -obs_shape: np.array
            Shape of an observation.
        -hidden_layer: int
            number of dense layers in the policy network
        nb_units: int / str
            number of units in each layer. If 'custom', then the number of units starts at 64 in the first 
            layer and doubles at each layer.
        -model: str
            Name of the wanted sb3 model ('DQN' or 'PPO' for now).
        '''
        super().__init__()
        self.obs_shape = obs_shape
        self.model_name=model
        self.device = device
        self.wandb_on = wandb_on

        if nb_units == 'custom':
            policy_kwargs = dict(net_arch=[64*(2**(i)) for i in range(hidden_layers)])
        else : 
            policy_kwargs = dict(net_arch=[nb_units for i in range(hidden_layers)]) # customized network architecture

        if model=='DQN':
            self.model = DQN("MlpPolicy", env, learning_rate=.00025, buffer_size=10000, learning_starts=10, 
                             batch_size=128, gamma=0.001, target_update_interval=250, verbose=2, 
                             exploration_final_eps=0.1, policy_kwargs=policy_kwargs, seed=seed, device=device)
        elif model == 'PPO':
            self.model = PPO("MlpPolicy", env, learning_rate=.00025, n_steps=512, batch_size=128, n_epochs=10,
                             gamma=0.001, policy_kwargs=policy_kwargs, verbose=1, seed=seed, device=device)
        else:
            raise(ValueError("Model unknown"))
        
    def save(self, path):
        self.model.save(path)
    
    def load(self, path):
        if self.model_name=='DQN':
            self.model = DQN.load(path)
        elif self.model_name=='PPO':
            self.model = PPO.load(path)
        else:
            raise(ValueError("model unknown"))

    def set_training_mode(self, training=True):
        '''
        -training: bool
            True if the agent is training, else False
        '''
        self.model.policy.set_training_mode(training)

    def _prepare_obs(self, obs):
        '''
        Helper function to transform numpy observation obs into fitted tensor
        '''
        try:
            batch_obs = obs.reshape((-1,) + self.obs_shape)
        except ValueError:
            raise ValueError("Expected an observation of shape {} or {}, but received an observation of shape {}".format( self.obs_shape, (-1,)+self.obs_shape, obs.shape ) )

        if obs.shape == self.obs_shape:
            obs = torch.as_tensor(obs[None]).to(self.device)
        elif obs.shape == batch_obs.shape:
            obs = torch.as_tensor(obs).to(self.device)
        else:
            raise ValueError("Expected an observation of shape {} or {}, but received an observation of shape {}".format( self.obs_shape, (-1,)+self.obs_shape, obs.shape ) )
        return obs

    def learn(self, eval_env, n_envs=1, save_dir=None, num_epoch=10):
        '''
        Learning process. The agent is evaluated on the evaluation environment eval_env after each episode.
        - save_dir: str
            path to the evaluation logs (.npz format) 
        '''
        episode_length = self.model.env.get_attr('episode_length')[0]
        
        if self.wandb_on:
            customwandbcallback = CustomWandbCallback(training_env=self.model.env, eval_env=eval_env, eval_freq=max(episode_length // n_envs, 1), save_dir=save_dir)
            self.model.learn(total_timesteps=episode_length*num_epoch, callback=customwandbcallback, reset_num_timesteps=False, progress_bar=True)
        else:
            self.model.learn(total_timesteps=episode_length*num_epoch, reset_num_timesteps=False, progress_bar=True)

    def forward(self, obs, deterministic=True, grad=False):
        obs = self._prepare_obs(obs)
        action, value, logprob = self.model.policy(obs, deterministic=deterministic)
        if grad:
            return action, value, logprob
        else:
            return action.detach().cpu().numpy(), value.detach().cpu().numpy(), logprob.detach().cpu().numpy()
    
    def policy(self, obs, grad=False):
        '''
        Returns the probability ditribution for all actions, given observation obs (different from self.model.policy!)
        Useful to compute adversarial perturbations.
        '''
        obs = self._prepare_obs(obs)

        s = nn.Softmax(dim=1)
        if self.model_name=='DQN':
            action_distribution = self.model.q_net(obs)
            action_distribution = s(action_distribution)
        elif self.model_name=='PPO':
            action_distribution = self.model.get_distribution(obs)
        else:
            raise(ValueError("Model unknown"))
        
        if grad:
            return action_distribution
        else:
            return action_distribution.detach().cpu().numpy()
