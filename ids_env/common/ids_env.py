#########################################################
# Custom gym environment class for intrustion detection #
#########################################################
from typing import Tuple

import gym
import pandas as pd
import numpy as np

class CustomIDSEnv(gym.Env):
  """
  Custom IDS Environment that follows gym interface.
  """

  def __init__(self, train_test : str, data_path : str, data='KDD', binary=False):
    '''
    Initializing the environment with the wanted dataset

    Parameters
    ----------
    train_test : str
      "train" or "test" phase, changes the sate selection process (random, adversarial, normal).
    data_path : str
      The path where the training or testing data can be found.
    data: str
      'KDD' or 'AWID', chosen dataset to work with
    binary: bool
      Classification mode : if binary=True, then two-class, else multi-class
    
    '''
    super(CustomIDSEnv, self).__init__()

    self.train_test = train_test
    self.binary=binary

    # Loading training or testing data
    self.df = pd.read_parquet(data_path)
    
    # Separating features and labels
    self.X = self.df.drop('labels', axis=1)
    self.y = self.df['labels'] # name of the labels
    if data=='KDD':
      self.attack_map = { 'normal': 'normal',
                        
                        'back': 'DoS',
                        'land': 'DoS',
                        'neptune': 'DoS',
                        'pod': 'DoS',
                        'smurf': 'DoS',
                        'teardrop': 'DoS',
                        'mailbomb': 'DoS',
                        'apache2': 'DoS',
                        'processtable': 'DoS',
                        'udpstorm': 'DoS',
                        
                        'ipsweep': 'Probe',
                        'nmap': 'Probe',
                        'portsweep': 'Probe',
                        'satan': 'Probe',
                        'mscan': 'Probe',
                        'saint': 'Probe',
                    
                        'ftp_write': 'R2L',
                        'guess_passwd': 'R2L',
                        'imap': 'R2L',
                        'multihop': 'R2L',
                        'phf': 'R2L',
                        'spy': 'R2L',
                        'warezclient': 'R2L',
                        'warezmaster': 'R2L',
                        'sendmail': 'R2L',
                        'named': 'R2L',
                        'snmpgetattack': 'R2L',
                        'snmpguess': 'R2L',
                        'xlock': 'R2L',
                        'xsnoop': 'R2L',
                        'worm': 'R2L',
                        
                        'buffer_overflow': 'U2R',
                        'loadmodule': 'U2R',
                        'perl': 'U2R',
                        'rootkit': 'U2R',
                        'httptunnel': 'U2R',
                        'ps': 'U2R',    
                        'sqlattack': 'U2R',
                        'xterm': 'U2R'
                    }
      self.y = self.y.replace(self.attack_map) # Categorical labels
    
    self.attack_types = sorted(list(pd.unique(self.y)))

    # Switch: put normal class first if it's not already the case
    idx_normal = self.attack_types.index('normal')
    self.attack_types[0], self.attack_types[idx_normal] = self.attack_types[idx_normal], self.attack_types[0]

    # Episode length depending on the train/test setting
    self.episode_length=self.X.shape[0]

    # Indexes of items in each attack type
    self.indexes = [[] for i in range(len(self.attack_types))]
    for i,att in enumerate(self.attack_types):
      self.indexes[i] = np.array(self.y.loc[self.y == att].index)

    # Action space
    n_actions = len(self.attack_types)
    self.action_space = gym.spaces.Discrete(n_actions)

    # Binary classification
    if self.binary==True:
      n_actions=2
      self.action_space=gym.spaces.Discrete(n_actions)
      self.binary_map = {key:'Attack' for key in self.attack_types}
      self.binary_map['normal']='Normal'
      self.y_bin=self.y.replace(self.binary_map)

    # Observation space
    self.observation_space = gym.spaces.Box(low=np.array(self.X.min(axis=0)), high=np.array(self.X.max(axis=0))) #shape is inferred from low and high

  
  def reset(self):
    #Initialize timestep, done, monitoring helpers
    self.num_step = 0
    self.done= False 
    self.total_rewards = 0

    # get index and label
    if self.train_test == 'train': # adversarial sampling : for now, choose one class at random for each state
      category = np.random.randint(len(self.attack_types))
      idx = np.random.choice(self.indexes[category])
      self.label =  np.sign(category) if self.binary else category
    else :
      idx = self.num_step
      self.label = np.sign(self.attack_types.index(self.y.iloc[idx])) if self.binary else self.attack_types.index(self.y.iloc[idx])

    # Get state from index
    self.state = np.array(self.X.iloc[idx], dtype='float32')

    return self.state

  def step(self, action : int) ->  Tuple[np.array, float, bool, dict]:
    '''
    Calculate reward and give transition infos
    '''
    self.num_step +=1
    self.action = action

    # Done ?
    self.done = (self.num_step>self.episode_length-2)

    # get the corresponding reward
    if self.action == self.label : 
      reward = 1.
    else : 
      reward = -1.

    self.total_rewards += reward

    # get next state and label
    if self.train_test == 'train': # adversarial sampling
      category = np.random.randint(len(self.attack_types))    # TODO add a sampling function (adversarial selector agent, etc.)
      idx = np.random.choice(self.indexes[category])
      self.label = np.sign(category) if self.binary else category
    else :
      idx = self.num_step
      self.label = np.sign(self.attack_types.index(self.y.iloc[idx])) if self.binary else self.attack_types.index(self.y.iloc[idx])
    self.state = np.array(self.X.iloc[idx], dtype='float32')

    # info ?
    info = {}

    return self.state, reward, self.done, info
  
  def render(self, mode='human'):
    '''
    For evaluation
    '''
    print("Num timesteps : ", self.num_timesteps)
    print("Total rewards : ", self.total_rewards)
    return True