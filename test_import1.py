#!/usr/bin/env python
# coding: utf-8

# In[7]:


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


# In[8]:


from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.adapt import MLkNN
import skmultilearn
from itertools import chain
import ast


# In[9]:


import test_import2


# In[10]:


import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
    
  
 #-------- Make Folder ------------------ 
path = os.path.join('./', 'Result')
if not os.path.exists(path):
    print('Make: ',path)
    os.mkdir(path)
else:
    print('Exist: ',path)
#------------------------------ 
path = os.path.join('./Result/', 'Graph_Data')
if not os.path.exists(path):
    print('Make: ',path)
    os.mkdir(path)
else:
    print('Exist: ',path)  
#------------------------------ 
path = os.path.join('./Result/', 'Site_DataSet')
if not os.path.exists(path):
    print('Make: ',path)
    os.mkdir(path)
else:
    print('Exist: ',path)


# In[11]:


if torch.cuda.is_available():
    torch.device("cuda") 
    #n_gpu = torch.cuda.device_count()
    print("torch.device : cuda")
    print(torch.cuda.get_device_name(0))
else:    
    torch.device("cpu")
    n_gpu = 0;
    print("torch.device : cpu")
    


# In[12]:


n_gpu = 1    #   0: "cpu"   &    1: "cuda"


# In[13]:


some_words_in_one_Text = 3000        # entekhab jomlati ke teaadade kalamatesh mosavi ya kam tar az meghdar taain shodeh bashad
some_Text_for_train_and_test = 2679   #teadade jolamat ya post baraye train + validation + test 


# In[14]:


epochs= 20

Header_Files = 'Sentiment2L_MultiLabel'
#Header_Files = 'sp_check'

Site_FileName = 'Site_xxxxx'

Flag_BinaryClassifiction = False
Label_Sort_Reverse = True

valid_size = 0.1111
test_size = 0.1


truncating="post"   # post or pre
padding="post"      # post or pre


seq_len=150
BatchSize = 16
LRate = 0.0001
Random_State = 2018


# In[15]:



'''
# --------------------For Dataset Part ------------------------
Input_Folder = "./Data/" 
data = pd.read_csv(Input_Folder + "twitter_extract_DIARRA_Sentences.csv", encoding="utf-8").fillna('')

dataNext = pd.read_csv(Input_Folder + "twitter_extract_JOBERTHE_Sentences.csv", encoding="utf-8").fillna('')
data = data.append(dataNext,sort= False)


dataNext = pd.read_csv(Input_Folder + "twitter_extract_LAURIE_Sentences.csv", encoding="utf-8").fillna('')
data = data.append(dataNext,sort= False)


data.to_csv("./Data/temp.csv",sep=",", index=None)
data = pd.read_csv("./Data/temp.csv", encoding="utf-8").fillna(method="ffill")
'''

# --------------------------------------------------------------------
#data = pd.read_csv("./Data/Tweeter_Sentence_Girls7000.csv", encoding="utf-8").fillna('')
#data = pd.read_csv("./Data/sp_check_DataSet_Small.csv", encoding="utf-8").fillna('')

#data = pd.read_csv("./Data/2020_10_16/RS_Medvalgo_2020_10_16.csv", encoding="utf-8").fillna('')
#data = pd.read_csv("./Data/Tweeter_Adnan_Sentence.csv", encoding="utf-8").fillna('')
#data = pd.read_csv("./Data/Test/Test_sp_check.csv", encoding="utf-8").fillna('')

#data = pd.read_csv("./Data/2020-10-24/RS_Medvalgo_Sentence_2020-10-24.csv", encoding="utf-8").fillna('')
#data = pd.read_csv("./Data/2020-11-05/RS_Medvalgo_Sentence_2020-11-05.csv", encoding="utf-8").fillna('')
data = pd.read_csv("./Data/2020-11-05/RS_Medvalgo_Sentence_2020-11-05_Without_Repeat.csv", encoding="utf-8").fillna('')

#TextAll = data["Text"].values
#TagAll = data['Classes'].values
data


# In[16]:


### Clean UP ###

#remove non-ascii words and characters
data['Text'] = [''.join([i if ord(i) < 128 else '' for i in Text]) for Text in data['Text']]


data['Text'] = data['Text'].str.replace(r'\n', r' ')
data['Text'] = data['Text'].str.replace(r'\r', r'')
data['Text'] = data['Text'].str.replace(r'  ', r' ')

# remove same char (ex. replace : ??????  with  ? )  -------
data['Text']= data['Text'].str.replace(r'([\!?<>:;])\1{%d,}'%(1), r'\1')


# In[ ]:





# In[ ]:





# In[17]:


flag_Remain_Select_Label = True
# remove label haye amadeh dar remove_label_list

if flag_Remain_Select_Label:
    
    ## Emotion ##
    Emotion_label = ['Anger','Concern','Disappointment','Disgusted','Dissatisfaction','Fear','Happy','Hopeful',
                     'Sad','Satisfaction','Surprised',]
                         
      
    ## Topic All##
    Topic_label = ['Ads and Campaigns','Clinical Trial','Cost/Insurance/Access','Counterfeit Avastin','Covid-19',
                   'Diagnosis and Symptoms','Diet and Lifestyle','HCP','Immunology','Interactions with Doctors',
                   'Media','Off Label','Oncology','Ophthalmology','Patient', 'Potential Misinformation',
                   'Quality of Life', 'Rare disease', 'Scientist', 'Treatment Comparison', 'Treatment Outcomes',
                   'Treatment Side Effects', 'Treatment Switchover','Treatment efficacy', 'Unmet Medical Needs']
    
    ## Topic ##
    Topic_label_2 = ['Ads and Campaigns','Clinical Trial','Cost/Insurance/Access','Counterfeit Avastin','Covid-19',
                   'Diagnosis and Symptoms','Diet and Lifestyle','HCP',#'Immunology',
                    'Interactions with Doctors',
                   'Media','Off Label','Oncology','Ophthalmology','Patient', 'Potential Misinformation',
                   'Quality of Life', 'Rare disease', 'Scientist', 'Treatment Comparison', 'Treatment Outcomes',
                   'Treatment Side Effects', 'Treatment Switchover','Treatment efficacy' #, 'Unmet Medical Needs'
                    ]
    
    ## Sentiment 3L ##
    Sentiment3L_label = ['Negative','Neutral','Positive']
    
    ## Sentiment 2L ##
    Sentiment2L_label = ['Negative','Positive']
         

        
    # Set Remain_label_list  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  
    Remain_label_list = Sentiment3L_label    
    
    
    data = test_import2.Remain_Select_Label(data,Remain_label_list)


# In[18]:


print(data.head(10))


# In[ ]:





# In[ ]:




