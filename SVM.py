#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd
import numpy as np
import seaborn as sb

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
# the support vector machine class :  SVM  
from sklearn.svm import SVC 

# parsing and preprocessing :
import gmparser as parse


# In[31]:


data, encoded_data, X, y = parse.main()


# In[32]:


data.head()


# In[33]:


#X


# In[34]:


X2 = preprocessing.scale(X)


# In[35]:


#X2


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X2, y, random_state = 0) 


# In[37]:


kf = KFold(n_splits=5, shuffle=False).split(y)


# In[38]:


#print('Training', 7*'\t', 'Testing')
#for i, j in kf:
#    print(f'{i[0:4]}... \t{i[len(i)-5:len(i)]}, \t{j}')


# In[39]:


svm_model_linear  = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_model_rbf     = SVC(kernel = 'rbf', C = 1).fit(X_train, y_train) 
svm_model_sigmoid = SVC(kernel = 'sigmoid', C = 1).fit(X_train, y_train) 
svm_predictions   = svm_model_sigmoid.predict(X_test) 


# In[40]:


kernels = ['linear', 'rbf', 'sigmoid']
svc  = lambda x: SVC(kernel = x, C = 1, gamma = 'auto')
SVMs = dict(
    [(i, svc(i)) for i in kernels]
)


# In[41]:


SVMs


# In[14]:


sigmoid_cvl = cross_val_score(SVMs['sigmoid'], X2, y, scoring='accuracy', cv = 15)


# In[29]:


print(f'Cross validation scores: \nMin: {sigmoid_cvl.min()}, Mean: {round(sigmoid_cvl.mean(),3)}, Max: {sigmoid_cvl.max()}')


# In[16]:


sb.distplot(sigmoid_cvl)


# In[42]:


cross_vals = {}
for kernel in SVMs.keys():
     _tmp = cross_val_score(SVMs[kernel], X2, y, scoring='accuracy', cv = 15)
     cross_vals.update({kernel: _tmp})  


# In[46]:


for val in cross_vals.keys():
    print(f'{val} kerlnel, mean accuracy: {round(cross_vals[val].mean(), 3)}')


# ## Grid Search for optimizing parameters

# In[66]:


param_grid = {
    'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
    'C': [0.1, 1, 10, 100], 
    'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]
}


# In[67]:


# Make grid search classifier
clf_grid = GridSearchCV(SVC(), param_grid, verbose=1)


# In[68]:


clf_grid.fit(X_train, y_train)


# In[69]:


print("Best Parameters:\n", clf_grid.best_params_)
print("Best Estimators:\n", clf_grid.best_estimator_)


# In[70]:


optimum = cross_val_score(SVC(**clf_grid.best_params_) , X2, y, scoring='accuracy', cv = 15)


# In[71]:


print(f'Cross validation scores: \nMin: {optimum.min()}, Mean: {round(optimum.mean(),3)}, Max: {optimum.max()}')


# In[72]:


sb.distplot(sigmoid_cvl)


# In[65]:


help(GridSearchCV)


# In[ ]:




