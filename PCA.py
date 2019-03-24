#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random as rd

from sklearn.decomposition import PCA
from sklearn import preprocessing

import matplotlib.pyplot as plt
import seaborn as sb

import gmparser as parse


# In[2]:


data1, data2, X, y = parse.main()


# In[3]:


scaled_data = preprocessing.scale(X) 


# In[4]:


pca = PCA()


# In[5]:


pca.fit(scaled_data)


# In[6]:


pca_data = pca.transform(scaled_data)


# In[7]:


per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)


# In[8]:


# Create labels for the scree plot.
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)] 


# In[9]:


plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot') 


# In[ ]:




