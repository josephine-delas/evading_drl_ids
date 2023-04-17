#########################################################
# Utilitary functions for plotting and evaluation       #
#########################################################

import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


def print_stats(attack_types, y_true, y_pred):
    '''
    Prints and returns statistics for each attack category (true, estimated, correctly estimated)

    Parameters
    ----------
    -y_true: (int array)
        true attack category 
    -y_pred: (int array)
        predicted attack category

    Outputs
    -------
    -true_labels: int array
        number of items in each label category
    -estimated_labels: int array
        number of estimated items in each category
    -estimated_correct_labels: int array
        number of correctly estimated items in each category
    
    '''
    total_reward = 0    
    true_labels = np.zeros(len(attack_types),dtype=int) # number of items in each category
    estimated_labels = np.zeros(len(attack_types),dtype=int) # number of estimated items in each category
    estimated_correct_labels = np.zeros(len(attack_types),dtype=int) # number of correctly estimated items in each category


    types,counts = np.unique(y_true,return_counts=True)
    for i in range(len(types)):
        true_labels[types[i]] += counts[i]

    for i,a in enumerate(y_pred):
        estimated_labels[a] +=1              
        if a == y_true[i]:
            total_reward += 1
            estimated_correct_labels[a] += 1

    outputs_df = pd.DataFrame(index = attack_types,columns = ["Estimated","Correct","Total"])
    for indx,att in enumerate(attack_types):
        outputs_df.iloc[indx].Estimated = estimated_labels[indx]
        outputs_df.iloc[indx].Correct = estimated_correct_labels[indx]
        outputs_df.iloc[indx].Total = true_labels[indx]

    print(outputs_df)
    return true_labels, estimated_labels, estimated_correct_labels

def calcul_rates(y_true, y_pred):
  '''
  Gives false positive and false negative rates for the normal class
  y_true and y_pred are int arrays 
  '''
  conf_mat = metrics.confusion_matrix(y_true, y_pred, labels = range(len(np.unique(y_true))))
  FP = np.sum(conf_mat[1:,0])
  TN = np.sum(conf_mat[1:,1:])
  FN = np.sum(conf_mat[0, 1:])
  TP = conf_mat[0,0]

  FPR = FP/(FP+TN) # Sur toutes les attaques combien sont catégorisées comme normales
  FNR = FN/(FN+TP) # Sur tous les normaux combien sont catégorisés comme attaques

  return(FPR, FNR)


def plot(nb_class, estimated_correct_labels, true_labels, estimated_labels, attack_types, filepath):
  '''
  Plots  the outputs given by print_stats()
  '''
  plt.rcParams['font.family'] = 'sans-serif'

  false_negative = np.abs(estimated_correct_labels-true_labels)
  false_positive = np.abs(estimated_correct_labels-estimated_labels)


  fig, ax = plt.subplots(figsize=(7,8))
  width = 0.2
  pos = np.arange(nb_class)
  p1 = plt.bar(pos, estimated_correct_labels, width, color='#228176ff')
  p3 = plt.bar(pos+width, false_positive, width, color='#e45c3aff') #false positive
  p2 = plt.bar(pos+width, false_negative, width, bottom=false_positive, color='#ffb85bff')#f6bf51ff') #false negative
  

  ax.yaxis.set_tick_params(labelsize=15)
  ax.set_xticks(pos+width/2)
  ax.set_xticklabels(attack_types,rotation='horizontal',fontsize = 'xx-large')
  ax.set_ylim(0, 11000)

  plt.legend(('True Positive','False Positive','False Negative'),fontsize = 'xx-large')
  
  plt.savefig(filepath, format='pdf', dpi=1000)
