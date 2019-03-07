#!/usr/bin/env python
# coding: utf-8

# In[2]:


# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
# the support vector machine class :  SVM  
from sklearn.svm import SVC 

# parsing and preprocessing :
import gmparser as parse


# In[3]:


X, y = parse.main()


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0) 


# In[5]:


svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 


# In[6]:


# model accuracy for X_test 
accuracy = svm_model_linear.score(X_test, y_test) 

# creating a confusion matrix 
cm = confusion_matrix(y_test, svm_predictions) 


# In[7]:


accuracy


# In[8]:


cm


# In[ ]:




