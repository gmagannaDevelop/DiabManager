#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
# the support vector machine class :  SVM  
from sklearn.svm import SVC 

# parsing and preprocessing :
import gmparser as parse


# In[50]:


data, encoded_data, X, y = parse.main()


# In[51]:


X2 = preprocessing.scale(X)


# In[80]:


X_train, X_test, y_train, y_test = train_test_split(X2, y, random_state = 0) 


# In[81]:


svm_model_linear  = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_model_rbf     = SVC(kernel = 'rbf', C = 1).fit(X_train, y_train) 
svm_model_sigmoid = SVC(kernel = 'sigmoid', C = 1).fit(X_train, y_train) 
svm_predictions   = svm_model_linear.predict(X_test) 


# In[82]:


# model accuracy for X_test 
linear_accuracy = svm_model_linear.score(X_test, y_test) 
sbf_accuracy = svm_model_rbf.score(X_test, y_test) 
sigmoid_accuracy = svm_model_sigmoid.score(X_test, y_test) 


# creating a confusion matrix 
cm = confusion_matrix(y_test, svm_predictions) 


# In[83]:


linear_accuracy


# In[84]:


sbf_accuracy


# In[85]:


sigmoid_accuracy


# In[66]:


cm


# In[67]:


data


# In[33]:


#scaled_data = preprocessing.scale(data)
plot = pd.core.frame.DataFrame(index=data['postp tag'])
#for i in data:
#    plot[i] = data[i]
plot['lol'] = list(data['activeInsulin'])


# In[35]:


plot.head()


# In[ ]:




