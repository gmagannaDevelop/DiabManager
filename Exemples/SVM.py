#!/usr/bin/env python
# coding: utf-8

# # Example SVM
# link : https://www.geeksforgeeks.org/multiclass-classification-using-scikit-learn/

# In[1]:


# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 


# In[2]:


# loading the iris dataset 
iris = datasets.load_iris()


# In[3]:


# X -> features, y -> label 
X = iris.data 
y = iris.target 

# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0) 


# In[4]:


# training a linear SVM classifier 
from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 


# In[5]:


# model accuracy for X_test 
accuracy = svm_model_linear.score(X_test, y_test) 

# creating a confusion matrix 
cm = confusion_matrix(y_test, svm_predictions) 


# In[6]:


accuracy


# In[7]:


cm


# In[8]:


iris.target


# In[9]:


y


# In[13]:


iris.data


# In[14]:


iris.target


# In[ ]:




