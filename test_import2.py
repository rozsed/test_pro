#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import os
import pandas as pd
import numpy as np
import keras
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils.data import TensorDataset
#from keras.preprocessing import text, sequence
#import time

from torch.optim import Adam
from datetime import datetime

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
#from seqeval.metrics import precision_score, recall_score, f1_score, classification_report,accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report,accuracy_score
from sklearn.metrics import auc,roc_auc_score,hamming_loss,roc_curve
from transformers import *
import random
import pickle

#from sklearn.metrics import roc_auc_score
import colorama
from colorama import Fore, Style
from torch.autograd import Variable
#import torch
#from torch import nn
from random import shuffle
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.metrics import confusion_matrix
from numpy import sqrt
from numpy import argmax
from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve
from scipy import interp


# In[2]:


from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.adapt import MLkNN
import skmultilearn
from itertools import chain
import ast


# In[3]:


def Remain_Select_Label(dataSet,label_list):
    for ii in range(len(dataSet)):
        aa = ast.literal_eval(dataSet['Classes'][ii])
        bb = aa.copy()
        for lab in aa:
            if lab not in label_list:           
                bb.remove(str(lab))   
        dataSet['Classes'][ii] = str(bb)
    return dataSet


# In[ ]:





# In[ ]:




